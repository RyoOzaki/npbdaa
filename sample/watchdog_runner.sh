#!/bin/bash

killtree() {
    local _pid=$1
    local _sig=${2:-KILL}
    kill -stop ${_pid} # needed to stop quickly forking parent from producing children between child killing and parent killing
    for _child in $(ps -o pid --no-headers --ppid ${_pid}); do
        killtree ${_child} ${_sig}
    done
    kill -${_sig} ${_pid}
}

INTERVAL=300 # sec
TARGET=log.txt # target file of watch
CONTINUE=false # flag of run runner.sh or continue.sh
WATCHDOG_LOG=watchdog_log.txt

label=sample_results
begin=1
end=20

while getopts cl:b:e: OPT
do
  case $OPT in
    "c" ) CONTINUE=true ;;
    "l" ) label="${OPTARG}" ;;
    "b" ) begin="${OPTARG}" ;;
    "e" ) end="${OPTARG}" ;;
  esac
done

echo "start-up watchdog" >> ${WATCHDOG_LOG}
echo "--ITERVAL=${INTERVAL}" >> ${WATCHDOG_LOG}
echo "--TARGET=${TARGET}" >> ${WATCHDOG_LOG}
if "${CONTINUE}" ; then
  echo "--CONTINUE mode" >> ${WATCHDOG_LOG}
else
  echo "--label=${label}" >> ${WATCHDOG_LOG}
  echo "--begin=${begin}" >> ${WATCHDOG_LOG}
  echo "--end=${end}" >> ${WATCHDOG_LOG}
fi

touch ${TARGET}

last=`ls --full-time ${TARGET} | awk '{print $6"-"$7}'`

echo -n "start process..." >> ${WATCHDOG_LOG}
if "${CONTINUE}" ; then
  sh continue.sh &
else
  sh runner.sh -l ${label} -b ${begin} -e ${end} &
fi
PID=$!
echo "done!" >> ${WATCHDOG_LOG}
echo "PID=${PID}" >> ${WATCHDOG_LOG}

trap 'echo "shutdown watchdog" >> ${WATCHDOG_LOG}; killtree ${PID}; exit 1'  1 2 3 15

sleep ${INTERVAL}

while [ `ps -a | grep "${PID}" -o` ] ; do

  current=`ls --full-time ${TARGET} | awk '{print $6"-"$7}'`

  if [ ${last} != ${current} ] ; then
    last=$current
  else
    echo -n "shutdown process ${PID}..." >> ${WATCHDOG_LOG}
    killtree ${PID}
    echo "done!" >> ${WATCHDOG_LOG}
    echo -n "rebooting process..." >> ${WATCHDOG_LOG}
    sh continue.sh &
    PID=$!
    echo "done!" >> ${WATCHDOG_LOG}
    echo "PID=${PID}" >> ${WATCHDOG_LOG}
  fi

  sleep ${INTERVAL}
done

echo "all process finished!" >> ${WATCHDOG_LOG}

exit 0