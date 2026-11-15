import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from evaluation.evaluate import run_evaluation
from tests.test_sam2_wrapper_fake import build_fake
from datasets.btcv_dataset import collate_btcv, BTCVSample


class TinyEvalSet(Dataset):
    def __len__(self): return 1
    def __getitem__(self, index):
        image = torch.rand(1, 1, 64, 64)
        y = torch.zeros(1, 64, 64, dtype=torch.long)
        y[:, 16:44, 18:42] = 1
        label = torch.nn.functional.one_hot(y, num_classes=2).permute(0, 3, 1, 2).float()
        return BTCVSample(image, label, "img0001", [9], torch.ones(1, dtype=torch.bool))


def test_evaluation_exports_prediction_with_case_slice_and_organ(tmp_path):
    torch.manual_seed(55)
    model = build_fake()
    loader = DataLoader(TinyEvalSet(), batch_size=1, collate_fn=collate_btcv)
    result = run_evaluation(model, loader, "cpu", k_infer=2,
                            compute_surface_distances=False, max_batches=1,
                            prediction_dir=tmp_path)
    files = list(tmp_path.glob("*.npz"))
    assert len(files) == 1
    with np.load(files[0], allow_pickle=False) as data:
        assert str(data["case_id"].item()) == "img0001"
        assert int(data["slice_index"].item()) == 9
        assert int(data["organ_class"].item()) == 1
        assert data["prediction"].shape == (64, 64)
        assert data["mean_probability"].dtype == np.float16
    assert result["protocol"]["prediction_export_dir"] == str(tmp_path)
