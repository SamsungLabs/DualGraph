import asyncio
import gc
import logging
import math
import time
from pathlib import Path
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd
import torch
import tqdm.asyncio
from graspologic import embed
from graspologic.utils import largest_connected_component
from rdflib import Graph as RDFGraph
from rdflib import Literal as RDFLiteral
from rdflib import URIRef
from sklearn.cluster import KMeans
from torch.nn.functional import normalize

from .base import LOGGER
from .utils import build_conversation

L = logging.getLogger(LOGGER)

MAX_BUCKET_SIZE = 100000


async def do_alignment(*, tkg_entities, llm_func, embedder, working_dir):
    skg_embedder_params = {
        "dimensions": 300,
        "num_walks": 40,
        "walk_length": 40,
        "window_size": 2,
        "iterations": 30,
        "random_seed": 1234,
    }
    align_config = {
        "num_epochs": 3000,
        "batch_size": 112,
        "temperature": 1.2,
        "embedding_dim_out": 245,
        "optimizer": "Adam",
        "lr": 5e-5,
        "sim_threshold": 0.6,
        "k": 3,
        "mode": "marriage",
    }

    embeddings = []
    node_keys = []

    rdf_g = RDFGraph()
    rdf_g.parse(working_dir / "skg.nt", format="nt")
    g = custom_rdflib_to_networkx_digraph(rdf_g)
    g = largest_connected_component(g)
    g_embeddings, g_node_keys = embed.node2vec_embed(g, **skg_embedder_params)  # type: ignore
    embeddings.append(g_embeddings)
    node_keys.append(g_node_keys)

    g_embeddings = await embedder(tkg_entities["description"].to_list())
    g_node_keys = tkg_entities["rdfox_id"].to_list()
    embeddings.append(g_embeddings)
    node_keys.append(g_node_keys)

    L.info("SKG embeddings shape is %s", embeddings[0].shape)
    L.info("TKG embeddings shape is %s", embeddings[1].shape)

    aligner = Aligner(llm_func)
    aligner.split_data(embeddings, node_keys)
    aligner.train_aligner(align_config, working_dir)
    output_df = await aligner.apredict(
        align_config["sim_threshold"],  # type: ignore
        align_config["k"],  # type: ignore
        align_config["mode"],  # type: ignore
        device="cpu",
    )
    f = output_df.apply(
        lambda x: x["aligned"] is not None and not (x["in_left"] and x["in_right"]),
        axis=1,
    )
    aligned = output_df[f]
    # Left is SKG
    # Right is TKG
    return aligned


def custom_rdflib_to_networkx_digraph(rdf_graph):
    g = nx.DiGraph()

    for sub, pred, obj in rdf_graph:
        # Extract node names
        s_name = sub.fragment if isinstance(sub, URIRef) else str(sub)
        o_name = obj.fragment if isinstance(obj, URIRef) else str(obj)
        p_label = pred.fragment if isinstance(pred, URIRef) else str(pred)
        if isinstance(obj, RDFLiteral):
            continue

        # Add nodes with attributes preserving original URI
        g.add_node(s_name, original_uri=str(sub))
        g.add_node(o_name, original_uri=str(obj))

        # Add edge with predicate label
        g.add_edge(s_name, o_name, predicate=p_label, original_predicate=str(pred))

    return g


class Aligner:
    # pylint: disable=attribute-defined-outside-init
    def __init__(
        self,
        llm_func,
    ):
        self.global_below_threshold_count = 0
        self.llm_func = llm_func
        self.prompt = (
            "You will be given an entity as well as a list of candidate entities. Your task is to "
            "decide if any of the candidate entities are suitable counterparts to the original "
            "entity, and if so, select the best one. The candidate entities will be given in "
            "order of likeliness, with the most likely candidate first.\n"
            "Please answer by only providing the letter associated with your chosen answer.\n"
            "Original entity: {original_entity}\n"
            "Candidate entities:\n"
            "{choices}\n"
        )

    def split_data(
        self,
        embeddings: list[np.ndarray],
        node_keys: list[list[str]],
    ):
        L.info("Splitting the data...")
        left_emb, right_emb = embeddings
        left_nodes, right_nodes = node_keys
        self.left_df = pd.DataFrame({"node": left_nodes, "emb": left_emb.tolist()})
        self.right_df = pd.DataFrame({"node": right_nodes, "emb": right_emb.tolist()})
        L.info(
            "Shape of left_df: %s  Shape of right_df: %s",
            str(self.left_df.shape),
            str(self.right_df.shape),
        )
        self.common_core = set(self.left_df["node"]).intersection(
            set(self.right_df["node"])
        )
        L.info("Length of the common core is %d", len(self.common_core))
        L.info(
            "Which accounts for %f of all nodes",
            len(self.common_core)
            / (len(set(left_nodes)) + len(set(right_nodes)) - len(self.common_core)),
        )
        self.left_df["common"] = self.left_df["node"].isin(self.common_core)
        self.right_df["common"] = self.right_df["node"].isin(self.common_core)

        self.left_common = self.left_df[self.left_df["common"]].set_index("node")
        self.right_common = (
            self.right_df[self.right_df["node"].isin(self.common_core)]
            .set_index("node")
            .loc[self.left_common.index]
        )
        self.left_diff = self.left_df[~self.left_df["common"]].set_index("node")
        self.right_diff = self.right_df[~self.right_df["common"]].set_index("node")
        L.info(
            "Shape of left_diff: %s  Shape of right_diff: %s",
            str(self.left_diff.shape),
            str(self.right_diff.shape),
        )

        t = time.time()
        self.left_tensor_train = torch.tensor(self.left_common["emb"])
        self.right_tensor_train = torch.tensor(self.right_common["emb"])
        L.info("Tensor putting took: %fs", time.time() - t)

        # Create lists for each category of nodes
        t = time.time()
        common_rows = [
            {"node": node, "in_left": True, "in_right": True, "aligned": node}
            for node in self.common_core
        ]

        left_diff_rows = [
            {"node": node, "in_left": True, "in_right": False, "aligned": None}
            for node in self.left_diff.index
        ]

        right_diff_rows = [
            {"node": node, "in_left": False, "in_right": True, "aligned": None}
            for node in self.right_diff.index
        ]

        # Concatenate all rows at once
        self.output_df = pd.DataFrame(
            common_rows + left_diff_rows + right_diff_rows
        ).set_index("node")

        L.info("Output df creation took: %fs", time.time() - t)
        L.info("Data splitting done...")

    def train_aligner(
        self,
        align_config: dict,
        save_path: Path,
    ):
        num_epochs = align_config["num_epochs"]
        batch_size = align_config["batch_size"]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # hardcoding contrastive alignment for now
        self.model = SelfSupConAligner(
            self.left_tensor_train.shape[1],
            self.right_tensor_train.shape[1],
            align_config["embedding_dim_out"],
            align_config["temperature"],
            device,
        ).to(device)

        left_tensor_train = self.left_tensor_train.to(device)
        right_tensor_train = self.right_tensor_train.to(device)

        model_pass_func = supcon_pass_fun
        optimizer = getattr(torch.optim, align_config["optimizer"])(
            self.model.parameters(), lr=align_config["lr"]
        )
        L.info("Training on %s...", str(device))
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch_start in range(0, len(self.left_tensor_train), batch_size):
                batch1 = left_tensor_train[batch_start : batch_start + batch_size]
                batch2 = right_tensor_train[batch_start : batch_start + batch_size]
                loss = model_pass_func(self.model, batch1, batch2)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            L.info("Epoch %d/%d - Loss: %.4f", epoch + 1, num_epochs, epoch_loss)
        L.info("Aligner trained")
        torch.save(self.model, save_path / "self_sup_contrast_aligner.pt")
        L.info(
            "Aligner saved to disk at %s",
            str(save_path / "self_sup_contrast_aligner.pt"),
        )

    async def ask_llm_about_candidates(
        self,
        original_node: str,
        candidate_nodes: list[str],
    ) -> tuple[str, str | None]:
        if candidate_nodes is None:
            return (original_node, None)
        choices = ""
        choice_letters_list = []
        if len(candidate_nodes) > 10:
            L.warning("We have more than 10 candidates")
        for idx, entity in enumerate(candidate_nodes):
            choice_letter = chr(ord("A") + idx)
            choice_letters_list.append(choice_letter)
            choices += f"{choice_letter}. {entity}\n"
        choices += f"{chr(ord('A') + len(candidate_nodes) + 1)}. None of the above.\n"
        L.debug(
            "For node %s the candidate nodes are: %s", original_node, candidate_nodes
        )
        prompt = self.prompt.format_map(
            {
                "original_entity": original_node,
                "choices": choices,
            }
        )

        completion, _ = await self.llm_func(build_conversation(prompt=prompt))
        if completion is not None:
            if completion[0] in choice_letters_list:
                L.info(
                    "%s aligned with %s",
                    original_node,
                    candidate_nodes[choice_letters_list.index(completion[0])],
                )
                return (
                    original_node,
                    candidate_nodes[choice_letters_list.index(completion[0])],
                )
        else:
            L.warning("Chat completion was None")
        return (original_node, None)

    async def apredict(
        self,
        sim_threshold: float,
        k: int,
        mode: Literal["left", "marriage"],
        device,
    ):
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        L.info("Starting alignment prediction...")
        with torch.no_grad():
            left_tensor_predict = torch.tensor(self.left_diff["emb"]).to(device)
            right_tensor_predict = torch.tensor(self.right_diff["emb"]).to(device)
            self.model.to(device)
            left_emb = self.model.left_embed(left_tensor_predict)
            right_emb = self.model.right_embed(right_tensor_predict)
            torch.cuda.empty_cache()
            gc.collect()

        if mode == "left":
            cross_sim = cosine_sim(left_emb, right_emb)
            top_k = cross_sim.topk(k)
            coroutines = []
            for i, node in enumerate(self.left_diff.index):
                top_k_values = top_k.values[i]
                top_k_indices = top_k.indices[i]
                filtered_indicies = top_k_indices[top_k_values > sim_threshold]
                if len(filtered_indicies) == 0:
                    L.info("No candidates above similarity threshold for %s", node)
                    self.global_below_threshold_count += 1
                    candidate_nodes = None
                else:
                    candidate_nodes = self.right_diff.index.values[filtered_indicies]
                    if isinstance(candidate_nodes, str):
                        candidate_nodes = [candidate_nodes]
                coroutine = self.ask_llm_about_candidates(node, candidate_nodes)  # type: ignore
                coroutines.append(coroutine)
            responses = await asyncio.gather(*coroutines)
            for node, aligned in responses:
                self.output_df.at[node, "aligned"] = aligned
            L.info(
                "There are %f of nodes below similarity threshold",
                self.global_below_threshold_count / len(responses),
            )
            L.info(
                "For %f of nodes llm found alignment",
                sum(1 for _, aligned in responses if aligned is not None)
                / len(responses),
            )

        elif mode == "marriage":
            ldf, rdf, cross_sim = bucketed_cosine_sim(
                left_emb, self.left_diff.index, right_emb, self.right_diff.index
            )
            left_top_k = cross_sim.topk(k)
            right_top_k = cross_sim.T.topk(k)
            left_coroutines = []
            for i, node in enumerate(ldf["node"]):
                top_k_values = left_top_k.values[i]
                top_k_indices = left_top_k.indices[i]
                filtered_indicies = top_k_indices[top_k_values > sim_threshold]
                if len(filtered_indicies) == 0:
                    self.global_below_threshold_count += 1
                else:
                    candidate_nodes = rdf.iloc[filtered_indicies][
                        "node"
                    ].values.tolist()
                    coroutine = self.ask_llm_about_candidates(node, candidate_nodes)
                    left_coroutines.append(coroutine)

            right_coroutines = []
            for i, node in enumerate(rdf["node"]):
                top_k_values = right_top_k.values[i]
                top_k_indices = right_top_k.indices[i]
                filtered_indicies = top_k_indices[top_k_values > sim_threshold]
                if len(filtered_indicies) == 0:
                    self.global_below_threshold_count += 1
                else:
                    candidate_nodes = ldf.iloc[filtered_indicies][
                        "node"
                    ].values.tolist()
                    coroutine = self.ask_llm_about_candidates(node, candidate_nodes)
                    right_coroutines.append(coroutine)

            left_responses = await tqdm.asyncio.tqdm.gather(
                *left_coroutines, desc="Left"
            )
            right_responses = await tqdm.asyncio.tqdm.gather(
                *right_coroutines, desc="Right"
            )
            # left_responses = await asyncio.gather(*left_coroutines)
            # right_responses = await asyncio.gather(*right_coroutines)
            marriage_count = 0
            L.info("Starting marriage alignment...")
            for l_node, l_alig in left_responses:
                for r_node, r_alig in right_responses:
                    if l_alig is not None and r_alig is not None:
                        if l_node == r_alig and r_node == l_alig:
                            L.info("Marriage aligned %s with %s", l_node, l_alig)
                            self.output_df.at[l_node, "aligned"] = l_alig
                            marriage_count += 1
            L.info("Finished marriage alignment.")
            L.info(
                "There are %f of nodes below similarity threshold",
                self.global_below_threshold_count
                / (len(left_responses) + len(right_responses)),
            )
            L.info(
                "For %f of nodes llm found alignment",
                marriage_count / (len(left_responses) + len(right_responses)),
            )
        else:
            raise ValueError(
                "Incorrect mode selected! Please choose either `left` or `marriage`!"
            )

        L.info("Alignment prediction done")
        return self.output_df


class SelfSupConAligner(torch.nn.Module):
    def __init__(
        self,
        embedding_dim_in_left,
        embedding_dim_in_right,
        embedding_dim_out,
        temperature,
        device,
    ):
        super().__init__()
        self.temperature = temperature
        self.left_embedder = torch.nn.Linear(
            embedding_dim_in_left, embedding_dim_out
        ).to(device)
        self.right_embedder = torch.nn.Linear(
            embedding_dim_in_right, embedding_dim_out
        ).to(device)
        self.device = device

    def forward(self, left, right):
        l_emb = self.left_embedder(left)
        r_emb = self.right_embedder(right)
        return l_emb, r_emb

    def loss_fun(self, left_emb, right_emb):
        left_emb = torch.nn.functional.normalize(left_emb, dim=1)
        right_emb = torch.nn.functional.normalize(right_emb, dim=1)
        sim = left_emb @ right_emb.T
        self_sim = torch.exp(sim.diag() / self.temperature)
        other_mask = abs(torch.eye(sim.shape[0]).to(self.device) - 1)
        other_sim = torch.exp((sim * other_mask) / self.temperature).sum(dim=1)
        example_loss = torch.log(self_sim / other_sim)
        batch_loss = -1 * example_loss.sum()
        return batch_loss

    def left_embed(self, left):
        return self.left_embedder(left)

    def right_embed(self, right):
        return self.right_embedder(right)


def supcon_pass_fun(model, batch1, batch2):
    out_l, out_r = model(batch1, batch2)
    loss = model.loss_fun(out_l, out_r)
    return loss


def cosine_sim(lemb, remb):
    return normalize(lemb, dim=1) @ normalize(remb, dim=1).T


def bucketed_cosine_sim(lemb: torch.Tensor, lindex, remb: torch.Tensor, rindex):
    lemb_numpy = lemb.numpy()
    remb_numpy = remb.numpy()
    tmp = [lemb_numpy, remb_numpy]
    embs = np.concatenate([lemb_numpy, remb_numpy], axis=0)
    l = (
        pd.DataFrame({"node": lindex, "label": "left"})
        .reset_index()
        .set_index(["index", "label"])
    )
    r = (
        pd.DataFrame({"node": rindex, "label": "right"})
        .reset_index()
        .set_index(["index", "label"])
    )
    df = pd.concat([l, r], axis=0, ignore_index=False)
    L.info("Starting the BUCKETEER!")
    b_ass = split_into_buckets(
        df,
        embs,
        MAX_BUCKET_SIZE,
    )
    df["bucket_assignment"] = b_ass
    valid_nodes = (
        df.reset_index()
        .groupby("bucket_assignment")
        .filter(lambda x: {"left", "right"}.issubset(set(x["label"])))
        .set_index(["index", "label"])
    )
    valid_numpy = []
    valid_dfs = []
    for label, emb_numpy in zip(["left", "right"], tmp):
        x = valid_nodes.reset_index(level=1)["label"] == label
        valid_numpy.append(emb_numpy[x.where(x).dropna().index])
        i = x.where(x).dropna().index.to_frame()
        i["label"] = label
        valid_dfs.append(valid_nodes.loc[i.set_index(["index", "label"]).index])
    valid_lemb = torch.from_numpy(valid_numpy[0])
    valid_remb = torch.from_numpy(valid_numpy[1])
    valid_ldf = valid_dfs[0]
    valid_rdf = valid_dfs[1]
    L.info("Starting the COSINE SIMILARITY CALCULATION!")
    sim = cosine_sim(valid_lemb, valid_remb)
    return valid_ldf, valid_rdf, sim


def split_into_buckets(
    base_df: pd.DataFrame,
    embeddings: np.ndarray,
    max_bucket_size: int,
) -> list[np.ndarray]:
    n = base_df.shape[0]
    if n < max_bucket_size or max_bucket_size < 0:
        num_buckets = 2
    else:
        num_buckets = 2 * math.ceil(n / max_bucket_size)
    start = time.perf_counter()
    clusterer = KMeans(n_clusters=num_buckets)
    labels = clusterer.fit_predict(embeddings)
    L.info("K-means constrained clustering took %fs", time.perf_counter() - start)
    return labels
