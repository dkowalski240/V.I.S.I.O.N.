"""Optional: the REAL models with their trained weights (downloads them; uses the GPU if present).

    set ASSISTANT_MODEL_TESTS=1
    pytest tests\\test_real_models.py -v

Uses scikit-image's sample astronaut photo (pip install scikit-image).
"""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("ASSISTANT_MODEL_TESTS") != "1",
                                reason="set ASSISTANT_MODEL_TESTS=1 to run the real models")


@pytest.fixture(scope="module")
def setup():
    cv2 = pytest.importorskip("cv2")
    torch = pytest.importorskip("torch")
    skdata = pytest.importorskip("skimage.data")
    from assistant.config import Config, apply_cpu_fallback

    cfg = Config()
    if not torch.cuda.is_available():
        apply_cpu_fallback(cfg)
    photo = cv2.cvtColor(skdata.astronaut(), cv2.COLOR_RGB2BGR)
    frame = np.full((720, 1280, 3), 40, np.uint8)
    frame[100:612, 380:892] = photo
    return cfg, frame


def test_faces_detect_align_embed_and_match(setup):
    cfg, frame = setup
    from assistant.faces.alignment import align_face
    from assistant.faces.detection import create_detector
    from assistant.faces.embedding import FaceNetEmbedder

    det = create_detector(cfg.faces.detection)
    faces = det.detect(frame)
    assert len(faces) == 1
    emb = FaceNetEmbedder(cfg.faces.embedding)
    crop = align_face(frame, faces[0], cfg.faces.alignment).image
    a, b = emb.embed([crop, crop[:, ::-1].copy()])  # the face and its mirror image
    assert float(a @ b) > cfg.faces.matching.t_match


def test_owlv2_finds_a_person_and_clip_embeds(setup):
    cfg, frame = setup
    from assistant.objects.detector import OWLv2Detector
    from assistant.objects.embedder import ClipEmbedder, crop_with_margin

    o = cfg.objects
    det = OWLv2Detector(o.owlv2_model, o.device, o.fp16, 0.1)
    found = det.detect(frame, ["a person", "a flag", "a laptop"])
    assert any(d.label == "a person" for d in found)
    person = [d for d in found if d.label == "a person"][0]
    assert 380 - 20 <= person.box[0] and person.box[2] <= 892 + 20  # inside the pasted photo
    clip = ClipEmbedder(o.embedder_model, o.embedder_pretrained, o.device, o.fp16)
    z = clip.embed([crop_with_margin(frame, person.box)] * 2)
    assert z.shape[0] == 2 and abs(float(z[0] @ z[1]) - 1.0) < 1e-3
