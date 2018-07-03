#!/bin/sh

label=sample_results

for dir in `ls ${label}`
do
  echo ${dir}

  rm -f summary_files/*
  rm -f results/*
  rm -f log.txt

  cp ${label}/${dir}/log.txt ./
  cp ${label}/${dir}/results/* results/
  cp ${label}/${dir}/summary_files/* summary_files/

  python summary_and_plot_light.py

  cp -r summary_files/ ${label}/${dir}/
  cp -r figures/ ${label}/${dir}/

done
