import os
import shortuuid
from prometheus_client import Gauge
from dataclasses import dataclass


@dataclass
class _MetricLabels:
  """All metrics emmitted should be configured with the following labels"""

  hostname: str = os.getenv("HOSTNAME", shortuuid.uuid())
  idx: int = 0


def _create_labeled_metric(metric, labels: _MetricLabels):
  return metric.labels(hostname=labels.hostname, idx=labels.idx)


class JetstreamMetricsCollector:
  """Wrapper class should be used to assure all metrics have proper tags"""

  def __new__(cls):
    if not hasattr(cls, "instance"):
      cls.instance = super(JetstreamMetricsCollector, cls).__new__(cls)
    return cls.instance

  # Metric definitions
  _prefill_backlog = Gauge(
      name="jetstream_prefill_backlog_size",
      documentation="Size of prefill queue",
      labelnames=["hostname"],
  )
  _slots_available_percentage = Gauge(
      name="jetstream_slots_available_percentage",
      documentation="The percentage of available slots in decode batch",
      labelnames=["hostname", "idx"],
  )

  def get_prefill_backlog_metric(self):
    return _create_labeled_metric(self._prefill_backlog, _MetricLabels())

  def get_slots_available_percentage_metric(self, idx: int):
    return _create_labeled_metric(
        self._slots_available_percentage,
        _MetricLabels(idx=idx),
    )
