#!/bin/bash

label=sample_results

while getopts l: OPT
do
  case $OPT in
    "l" ) label="${OPTARG}" ;;
  esac
done

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

rm -f summary_files/*
rm -f results/*
rm -f log.txt
