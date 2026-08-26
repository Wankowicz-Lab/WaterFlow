"""Tests for scripts/predict_waters.py -- end-to-end water prediction.

Unit tests cover the pure pieces (selection, model build, checkpoint load, path
collection, frame recovery). The integration test runs the whole pipeline with
tiny untrained gvp models, so it needs no trained checkpoints or embeddings.
"""

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.predict_waters import (
    _check_embeddings,
    _collect_struc_paths,
    _input_frame,
    load_checkpoint,
    parse_args,
    predict_structures,
    resolve_ckpt_dir,
    select_waters,
)
from src.confidence import build_confidence_model, ConfidenceGVP
from src.dataset import parse_asu_with_biotite
from src.flow import FlowMatcher, FlowWaterGVP
from src.inference_graph import build_inference_graph
from src.structure_io import read_space_group


@pytest.mark.unit
class TestSelectWaters:
    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            select_waters(torch.zeros(1, 3), torch.ones(1), mode="bogus")

    def test_density_keeps_top_n_by_confidence(self):
        # Four far-apart candidates: four singleton clusters, so the kept count
        # comes from the density formula alone. floor(0.6 * 5) = 3.
        pos = torch.tensor([[0.0, 0, 0], [10.0, 0, 0], [20.0, 0, 0], [30.0, 0, 0]])
        conf = torch.tensor([0.2, 0.9, 0.5, 0.7])
        sel_pos, sel_conf = select_waters(
            pos, conf, mode="density", density_ratio=0.6, num_asu_residues=5
        )
        assert sel_pos.shape[0] == 3
        assert torch.allclose(sel_conf, torch.tensor([0.9, 0.7, 0.5]))

    def test_density_has_no_cutoff(self):
        pos = torch.tensor([[0.0, 0, 0], [10.0, 0, 0]])
        conf = torch.tensor([0.9, 0.01])
        sel_pos, _ = select_waters(
            pos, conf, mode="density", density_ratio=1.0, num_asu_residues=2
        )
        assert sel_pos.shape[0] == 2

    def test_density_requires_ratio_and_residue_count(self):
        with pytest.raises(ValueError, match="density"):
            select_waters(torch.zeros(1, 3), torch.ones(1), mode="density")

    @pytest.mark.parametrize("ratio", [0.0, -1.0, float("nan"), float("inf")])
    def test_density_rejects_bad_ratio(self, ratio):
        with pytest.raises(ValueError, match="density_ratio"):
            select_waters(
                torch.zeros(1, 3),
                torch.ones(1),
                mode="density",
                density_ratio=ratio,
                num_asu_residues=5,
            )

    def test_nothing_survives(self):
        pos = torch.tensor([[0.0, 0, 0], [10.0, 0, 0]])
        conf = torch.tensor([0.9, 0.8])
        # confidence: threshold above every score
        sel_pos, sel_conf = select_waters(pos, conf, mode="confidence", threshold=1.0)
        assert sel_pos.shape == (0, 3) and sel_conf.shape == (0,)
        # density: floor(0.1 * 5) = 0
        sel_pos, sel_conf = select_waters(
            pos, conf, mode="density", density_ratio=0.1, num_asu_residues=5
        )
        assert sel_pos.shape == (0, 3) and sel_conf.shape == (0,)


@pytest.mark.unit
class TestSelectionCLI:
    """Each selection mode owns one knob and rejects the other's."""

    @staticmethod
    def _parse(*extra: str):
        base = ["--ckpt_dir", "c"]
        base += ["--struc", "s.pdb", "--out_dir", "o"]
        return parse_args(base + list(extra))

    def test_confidence_default_threshold(self):
        args = self._parse()
        assert args.confidence_threshold == 0.5 and args.density_ratio is None

    def test_density_default_ratio(self):
        args = self._parse("--selection", "density")
        assert args.density_ratio == 0.6 and args.confidence_threshold is None

    def test_confidence_rejects_density_ratio(self):
        with pytest.raises(SystemExit):
            self._parse("--density_ratio", "0.6")

    def test_density_rejects_threshold(self):
        with pytest.raises(SystemExit):
            self._parse("--selection", "density", "--confidence_threshold", "0.5")

    def test_threshold_range(self):
        with pytest.raises(SystemExit):
            self._parse("--confidence_threshold", "1.5")

    @pytest.mark.parametrize("ratio", ["0", "-1", "nan", "inf"])
    def test_density_ratio_range(self, ratio):
        with pytest.raises(SystemExit):
            self._parse("--selection", "density", "--density_ratio", ratio)


@pytest.mark.unit
class TestModelBuildAndLoad:
    def test_build_confidence_model(self):
        cfg = {"encoder_type": "gvp", "hidden_s": 64, "hidden_v": 8, "flow_layers": 1}
        model = build_confidence_model(cfg, torch.device("cpu"))
        assert isinstance(model, ConfidenceGVP)

    def test_load_round_trips(self, tmp_path):
        cfg = {"encoder_type": "gvp", "hidden_s": 64, "hidden_v": 8, "flow_layers": 1}
        m1 = build_confidence_model(cfg, torch.device("cpu"))
        ckpt = tmp_path / "best.pt"
        torch.save({"model_state_dict": m1.state_dict()}, ckpt)

        m2 = build_confidence_model(cfg, torch.device("cpu"))
        load_checkpoint(m2, ckpt, torch.device("cpu"))  # no raise
        assert not m2.training  # switched to eval

    def test_missing_checkpoint_raises(self, tmp_path):
        cfg = {"encoder_type": "gvp", "hidden_s": 64, "hidden_v": 8, "flow_layers": 1}
        model = build_confidence_model(cfg, torch.device("cpu"))
        with pytest.raises(FileNotFoundError):
            load_checkpoint(model, tmp_path / "nope.pt", torch.device("cpu"))

    def test_extra_checkpoint_keys_tolerated(self, tmp_path):
        # A checkpoint with tensors for a module the model no longer has (here a
        # second layer) loads fine: the surplus keys are dropped.
        base = {"encoder_type": "gvp", "hidden_s": 64, "hidden_v": 8}
        big = build_confidence_model({**base, "flow_layers": 2}, torch.device("cpu"))
        ckpt = tmp_path / "big.pt"
        torch.save({"model_state_dict": big.state_dict()}, ckpt)

        small = build_confidence_model({**base, "flow_layers": 1}, torch.device("cpu"))
        load_checkpoint(small, ckpt, torch.device("cpu"))  # no raise
        assert not small.training

    def test_missing_key_raises(self, tmp_path):
        # A checkpoint lacking a live parameter (here a second layer) would leave
        # it at init, so the load must fail loud rather than warn.
        base = {"encoder_type": "gvp", "hidden_s": 64, "hidden_v": 8}
        small = build_confidence_model({**base, "flow_layers": 1}, torch.device("cpu"))
        ckpt = tmp_path / "small.pt"
        torch.save({"model_state_dict": small.state_dict()}, ckpt)

        big = build_confidence_model({**base, "flow_layers": 2}, torch.device("cpu"))
        with pytest.raises(RuntimeError, match="missing from the checkpoint"):
            load_checkpoint(big, ckpt, torch.device("cpu"))


@pytest.mark.unit
class TestInputsAndFrame:
    def test_struc_paths(self, pdb_6eey):
        paths = _collect_struc_paths(SimpleNamespace(struc=[pdb_6eey], pdb_list=None))
        assert paths == [pdb_6eey]

    def test_duplicate_names_rejected(self):
        args = SimpleNamespace(struc=["a/protein.cif", "b/protein.cif"], pdb_list=None)
        with pytest.raises(ValueError, match="protein"):
            _collect_struc_paths(args)

    def test_pdb_list_resolves_names_with_and_without_ext(
        self, pdb_6eey, pdb_4h0b, tmp_path
    ):
        base = tmp_path / "pdbs"
        base.mkdir()
        shutil.copy(pdb_6eey, base)
        shutil.copy(pdb_4h0b, base)
        lst = tmp_path / "list.txt"
        # one entry carries an extension, one omits it; both resolve to a file
        lst.write_text("6eey_final.pdb\n4h0b_final\n")
        paths = _collect_struc_paths(
            SimpleNamespace(struc=None, pdb_list=str(lst), base_pdb_dir=str(base))
        )
        assert [Path(p).stem for p in paths] == ["6eey_final", "4h0b_final"]

    def test_pdb_list_warns_on_missing(self, tmp_path):
        lst = tmp_path / "list.txt"
        lst.write_text("does_not_exist\n")
        paths = _collect_struc_paths(
            SimpleNamespace(struc=None, pdb_list=str(lst), base_pdb_dir=str(tmp_path))
        )
        assert paths == []

    def test_input_frame(self, pdb_4h0b):
        kept, space_group = _input_frame(pdb_4h0b)
        protein, _w, lig = parse_asu_with_biotite(pdb_4h0b)
        assert int((kept.res_name == "HOH").sum()) == 0
        assert len(kept) == len(protein) + len(lig)
        assert space_group == "P 6"

    def test_graph_center_is_protein_centroid(self, pdb_4h0b):
        # predict_waters adds graph.center back to un-centre predictions, so it
        # must equal the ASU protein centroid the graph was built on.
        graph = build_inference_graph(pdb_4h0b, encoder_type="gvp")
        protein, _w, _lig = parse_asu_with_biotite(pdb_4h0b)
        assert graph.center.shape == (3,)
        assert np.allclose(graph.center.numpy(), protein.coord.mean(axis=0), atol=1e-4)


@pytest.mark.unit
class TestCheckEmbeddings:
    def test_gvp_skips_check(self):
        # gvp needs no embeddings, so nothing is required even with no cache.
        _check_embeddings(["/no/such/protein.cif"], "gvp", None)

    def test_missing_processed_dir_raises(self):
        with pytest.raises(ValueError, match="processed_dir"):
            _check_embeddings(["a/protein.cif"], "esm", None)

    def test_missing_esm_embedding_raises_naming_files(self, tmp_path):
        with pytest.raises(ValueError, match=r"protein\.cif.*other\.pdb"):
            _check_embeddings(["a/protein.cif", "b/other.pdb"], "esm", str(tmp_path))

    def test_present_esm_embeddings_pass(self, tmp_path):
        emb = tmp_path / "esm"
        emb.mkdir()
        (emb / "protein.pt").touch()
        _check_embeddings(["some/dir/protein.cif"], "esm", str(tmp_path))


@pytest.mark.unit
class TestResolveCkptDir:
    def test_explicit_dir_skips_detection(self, monkeypatch):
        def refuse(path):
            raise AssertionError("detection must not run for an explicit dir")

        monkeypatch.setattr("scripts.predict_waters.crystal_symmetry_check", refuse)
        assert resolve_ckpt_dir("my_ckpts", ["a.pdb"]) == Path("my_ckpts")

    def test_all_symmetric_picks_mates(self, monkeypatch):
        monkeypatch.setattr(
            "scripts.predict_waters.crystal_symmetry_check", lambda p: None
        )
        assert resolve_ckpt_dir(None, ["a.pdb", "b.pdb"]).name == "mates"

    def test_none_symmetric_falls_back_to_mates_off(self, monkeypatch):
        monkeypatch.setattr(
            "scripts.predict_waters.crystal_symmetry_check",
            lambda p: "no crystal symmetry record",
        )
        assert resolve_ckpt_dir(None, ["a.pdb", "b.pdb"]).name == "mates_off"

    def test_mixed_inputs_are_refused(self, monkeypatch):
        monkeypatch.setattr(
            "scripts.predict_waters.crystal_symmetry_check",
            lambda p: None if p == "good.pdb" else "no crystal symmetry record",
        )
        with pytest.raises(ValueError, match="mix"):
            resolve_ckpt_dir(None, ["good.pdb", "af.pdb"])


@pytest.mark.integration
class TestEndToEnd:
    @pytest.mark.parametrize("selection", ["confidence", "density"])
    def test_pipeline_writes_predicted_structure(
        self, selection, pdb_4h0b, gvp_encoder, tmp_path
    ):
        """Whole pipeline on tiny untrained gvp models: graph -> sample -> score ->
        cluster -> select -> un-center -> write. No checkpoints or embeddings."""
        device = torch.device("cpu")
        flow_model = FlowWaterGVP(
            encoder=gvp_encoder, hidden_dims=(64, 8), layers=1
        ).to(device)
        flow_matcher = FlowMatcher(model=flow_model, sampling_strategy="uniform_ball")
        conf_model = build_confidence_model(
            {"encoder_type": "gvp", "hidden_s": 64, "hidden_v": 8, "flow_layers": 1},
            device,
        )

        out_dir = tmp_path / "out"
        args = SimpleNamespace(
            processed_dir=None,
            include_mates=False,
            method="euler",
            num_steps=2,
            water_ratio=1.0,
            selection=selection,
            # Permissive settings so waters survive and the write path runs.
            confidence_threshold=0.0 if selection == "confidence" else None,
            density_ratio=1.0 if selection == "density" else None,
            out_dir=str(out_dir),
            out_format=".pdb",
        )

        predict_structures(
            [pdb_4h0b],
            flow_matcher,
            conf_model,
            {"encoder_type": "gvp"},
            args,
            device,
        )

        pdb_out = out_dir / "4h0b_final_pred.pdb"
        coords_out = out_dir / "4h0b_final_waters.txt"
        assert pdb_out.exists() and coords_out.exists()
        from biotite.structure.io.pdb import PDBFile

        pdb_file = PDBFile.read(str(pdb_out))
        written = pdb_file.get_structure(model=1)
        is_water = written.res_name == "HOH"
        n_waters = int(is_water.sum())
        assert n_waters > 0

        # Protein + ligand atoms are written unchanged, in the input frame, with
        # the input unit cell and space group.
        protein, _w, lig = parse_asu_with_biotite(pdb_4h0b)
        kept = protein + lig
        assert np.allclose(written.coord[~is_water], kept.coord, atol=1e-3)
        assert np.allclose(written.box, kept.box, atol=1e-3)
        assert read_space_group(str(pdb_out)) == read_space_group(pdb_4h0b)

        # Water rows in the txt match the written waters: x y z in the input
        # frame and confidence in [0, 1].
        rows = np.loadtxt(coords_out).reshape(-1, 4)
        assert rows.shape[0] == n_waters
        assert np.allclose(rows[:, :3], written.coord[is_water], atol=1e-3)
        assert ((rows[:, 3] >= 0) & (rows[:, 3] <= 1)).all()

    def test_predict_cache_writes_and_reuses(self, pdb_4h0b, gvp_encoder, tmp_path):
        """A second run reuses the cached graph and candidates, so its predicted
        waters are identical instead of freshly sampled."""
        device = torch.device("cpu")
        flow_model = FlowWaterGVP(
            encoder=gvp_encoder, hidden_dims=(64, 8), layers=1
        ).to(device)
        flow_model.eval()
        flow_matcher = FlowMatcher(model=flow_model, sampling_strategy="uniform_ball")
        conf_model = build_confidence_model(
            {"encoder_type": "gvp", "hidden_s": 64, "hidden_v": 8, "flow_layers": 1},
            device,
        )
        conf_model.eval()  # deterministic scores, so reuse yields identical output
        cache = tmp_path / "predict_cache"

        def run(out_dir):
            args = SimpleNamespace(
                processed_dir=None,
                include_mates=False,
                method="euler",
                num_steps=2,
                water_ratio=1.0,
                selection="confidence",
                confidence_threshold=0.0,
                density_ratio=None,
                out_dir=str(out_dir),
                out_format=".pdb",
            )
            predict_structures(
                [pdb_4h0b],
                flow_matcher,
                conf_model,
                {"encoder_type": "gvp"},
                args,
                device,
                cache=cache,
            )

        run(tmp_path / "out1")
        graph_pt = cache / "4h0b_final.pt"
        cand_pt = cache / "candidates" / "4h0b_final.pt"
        assert graph_pt.exists(), "flow-input graph not cached"
        assert cand_pt.exists(), "candidate waters not cached"
        assert "candidate_pos" in torch.load(cand_pt, weights_only=False)

        # Second run reuses the cached candidates -> identical predicted waters.
        run(tmp_path / "out2")
        r1 = np.loadtxt(tmp_path / "out1" / "4h0b_final_waters.txt").reshape(-1, 4)
        r2 = np.loadtxt(tmp_path / "out2" / "4h0b_final_waters.txt").reshape(-1, 4)
        assert r1.shape == r2.shape and np.allclose(r1, r2, atol=1e-4)
