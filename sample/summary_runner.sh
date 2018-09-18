#!/bin/bash

label=sample_results

while getopts l: OPT
do
  case $OPT in
    "l" ) label="${OPTARG}" ;;
  esac
done

result_dirs=`ls ${label} | grep "[0-9]*"`

mkdir -p summary_files
mkdir -p results

for dir in ${result_dirs}
do
  echo ${dir}

  sh clean.sh

  cp ${label}/${dir}/log.txt ./
  cp ${label}/${dir}/results/* results/
  cp ${label}/${dir}/summary_files/* summary_files/

  python summary_and_plot_light.py

  cp -r summary_files/ ${label}/${dir}/
  cp -r figures/ ${label}/${dir}/

done

sh clean.sh

python summary_summary.py ${label} ${result_dirs}

cp -r summary_files ${label}
cp -r figures ${label}

sh clean.sh
