#!/bin/sh

label=sample_results

mkdir -p ${label}

for i in `seq 1 20`
do
  echo ${i}

  i_str=$( printf '%02d' $i )
  rm -f results/*
  rm -f parameters/*
  rm -f summary_files/*
  rm -f log.txt

  python pyhlm_sample.py | tee log.txt

  mkdir -p ${label}/${i_str}/
  cp -r results/ ${label}/${i_str}/
  cp -r parameters/ ${label}/${i_str}/
  cp -r summary_files/ ${label}/${i_str}/
  cp log.txt ${label}/${i_str}/

done
