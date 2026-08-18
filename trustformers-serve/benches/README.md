# `trustformers-serve` benchmarks

## Removed in 0.2.1: `memory_pressure_regression.rs`

The memory-pressure regression benchmark measured nothing that existed. Every
figure in it was written into the file by hand:

* `bench_gpu_cleanup_handlers` picked a "memory freed" constant per GPU cleanup
  strategy (50 MiB for cache eviction, 200 MiB for model unloading, …) and then
  `std::thread::sleep`-ed for a hard-coded duration chosen to make each strategy
  look proportionally expensive. No GPU was involved, so the reported
  throughput was the ratio of two invented numbers.
* `bench_garbage_collection_handler` and `bench_buffer_compaction_handler`
  benchmarked `GarbageCollectionHandler` and `BufferCompactionHandler`, whose
  entire runtime *was* a `thread::sleep`, and whose "bytes freed" were
  hard-coded. Both handlers were removed in 0.2.1 for the same reason — see the
  note at the top of `src/memory_pressure/cleanup/handlers.rs`.
* `measure_cleanup_performance` fed a `mock_cpu_usage = 15.0` constant into
  `BaselineManager::record_baseline` and `check_regression`, so the recorded
  baselines carried a fabricated CPU figure, and `check_regression` could
  `panic!` a CI run over a "CPU usage increase" derived from that constant.
* `bench_error_handling_scenarios` used `fastrand` to invent a 10% failure rate
  and an 80% completion rate, and returned "5 MiB partial cleanup" / "15 MiB
  successful cleanup" constants.

A benchmark that sleeps for a chosen duration and reports a chosen byte count
does not measure the system; it publishes its own inputs as results, and the
baseline JSON it writes makes those inputs look like history. Benchmarking the
real handlers requires driving a real `CacheManager` or `ModelRegistry`, which
is what `src/memory_pressure/cleanup/handlers.rs` now takes.

`basic_benchmarks.rs` and `performance_benchmarks.rs` are unaffected.
