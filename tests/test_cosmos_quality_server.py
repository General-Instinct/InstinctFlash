import pytest

from benchmarks.vla.cosmos_quality_server import QualitySession
from benchmarks.vla.util import ConfigurationError


class Policy:
    def __init__(self):
        self.requests = []

    def infer(self, request):
        if request.get("fail"):
            raise ValueError("native generation failed")
        self.requests.append(request)
        return {"ack": len(self.requests)}


def test_reset_precedes_generation_and_preserves_wire_fields():
    policy = Policy()
    session = QualitySession(policy)
    assert session.infer({"reset": True, "episode_id": "smoke/0"}) == {"ack": 1}
    request = {"observation/image": b"camera", "request_id": 0,
               "benchmark_identity_sha256": "identity"}
    assert session.infer(request) == {"ack": 2}
    assert policy.requests[-1] is request


def test_reconnection_cannot_continue_previous_stream():
    policy = Policy()
    QualitySession(policy).infer({"reset": True})
    session = QualitySession(policy)
    with pytest.raises(ConfigurationError, match="reset first"):
        session.infer({"request_id": 1})
    assert len(policy.requests) == 1


def test_failed_request_cannot_be_retried_on_connection():
    policy = Policy()
    session = QualitySession(policy)
    session.infer({"reset": True})
    with pytest.raises(ValueError, match="native generation"):
        session.infer({"fail": True})
    with pytest.raises(ConfigurationError, match="cannot resume"):
        session.infer({"reset": True})
    assert len(policy.requests) == 1


@pytest.mark.parametrize("payload", [None, [], "reset", {"reset": 1}])
def test_malformed_reset_never_reaches_runtime(payload):
    policy = Policy()
    with pytest.raises(ConfigurationError):
        QualitySession(policy).infer(payload)
    assert not policy.requests
