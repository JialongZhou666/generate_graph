#!/usr/bin/env bash
DataSets="Cora"
Rates="0.05 0.10 0.15 0.20"
Methods="metattack"
Devices="cuda:5"
for dataset in $DataSets; do
  for rate in $Rates; do
    for method in $Methods; do
      for times in {1..1000}; do
        for device in $Devices; do
          python baseline_attacks.py --dataset $dataset --rate $rate --method $method --times $times --device $device
        done
      done
    done
  done
done
