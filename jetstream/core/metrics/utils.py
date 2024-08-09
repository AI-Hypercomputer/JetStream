from jetstream.core.orchestrator import ActiveRequest


def get_time_per_prefill_token(request: ActiveRequest, true_length: int):
  return (
      request.metadata.transfer_enqueue_time
      - request.metadata.prefill_dequeue_time
  ) / true_length


def get_queue_duration(request: ActiveRequest):
  return (
      # Time in prefill queue
      request.metadata.prefill_dequeue_time
      - request.metadata.prefill_enqueue_time
      # Time in transfer queue
      + request.metadata.transfer_dequeue_time
      - request.metadata.transfer_enqueue_time
      # Time in generate queue
      + request.metadata.generate_dequeue_time
      - request.metadata.generate_enqueue_time
  )


def get_tpot(request: ActiveRequest, result_tokens):
  return (
      request.metadata.complete_time - request.metadata.transfer_enqueue_time
  ) / result_tokens.get_result_at_slot(slot).lengths


def get_wait_time(request: ActiveRequest):
  total_time = request.metadata.complete_time - request.metadata.start_time
  prefill_time = (
      request.metadata.transfer_enqueue_time
      - request.metadata.prefill_dequeue_time
  )
  generate_time = (
      request.metadata.complete_time - request.metadata.generate_dequeue_time
  )
  return total_time - prefill_time - generate_time
