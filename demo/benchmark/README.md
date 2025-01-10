# Benchmark Readme

## Sample data creation

Sample data can be created with the script benchmark_data_prep.py. It expects a directory with .pkl.gz pointcloud files and turns it into .bin files.

## In order to generate the benchmark artifacts for the benchmark/sample data, one has to run following script via CLI:

```
python3 demo/benchmark/benchmark_tool.py /home/tstubler/testing_data/mmdet3d_test/model/fern3d_b1-b7-50000_v0 --output_path /home/tstubler --dataset_path /home/tstubler/testing_data/mmdet3d_test/workspace/points/
```