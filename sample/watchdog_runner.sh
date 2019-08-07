#!/bin/bash

INTERVAL=300 # sec
TARGET=log.txt # target file of watch
CONTINUE=false # flag of run runner.sh or continue.sh

while getopts cl:b:e: OPT
do
  case $OPT in
    "c" ) CONTINUE=true ;;
    "l" ) label="${OPTARG}" ;;
    "b" ) begin="${OPTARG}" ;;
    "e" ) end="${OPTARG}" ;;
  esac
done

touch ${TARGET}

last=`ls --full-time ${TARGET} | awk '{print $6"-"$7}'`

if [ ${CONTINUE} ] ; then
  sh continue.sh &
else
  sh runner.sh -l ${label} -b ${begin} -e ${end} &
fi
PID=$!

trap 'kill ${PID} ; exit 1'  1 2 3 15

sleep ${INTERVAL}

while [ `ps -a | grep "${PID}" -o` ] ; do

  current=`ls --full-time ${TARGET} | awk '{print $6"-"$7}'`

  if [ ${last} != ${current} ] ; then
    last=$current
  else
    kill ${PID}
    sh continue.sh &
    PID=$!
  fi

  sleep ${INTERVAL}
done

exit 0
