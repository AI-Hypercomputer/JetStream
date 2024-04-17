import argparse

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Benchmark the online serving throughput."
  )
  parser.add_argument(
      "--run-eval",
      type=bool,
      default=False,
      help="Whether to run evaluation script on the saved outputs",
  )
  parser.add_argument(
      "--warmup-first",
      type=bool,
      default=False,
      help="Whether to send warmup req first",
  )


  parsed_args = parser.parse_args()
  print(parsed_args)
