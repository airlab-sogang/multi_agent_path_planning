#!/bin/bash

# 여기는 건들지 말아주세요
execute_with_timeout() {
    local command_to_run="$1"
    # Run the command with a 5-minute timeout
    timeout 300 $command_to_run
    cmd_status=$?

    # Print messages with colors
    if [ $cmd_status -eq 124 ]; then
        echo -e "\033[0;31mTimeout! Command '$command_to_run' was forcibly terminated.\033[0m"
        echo "Timeout! Command '$command_to_run' was forcibly terminated." >> ecbs_result.txt
    elif [ $cmd_status -ne 0 ]; then
        echo -e "\033[0;31mUnexpected termination! Command '$command_to_run' exited with status $cmd_status.\033[0m"
        echo "Unexpected termination! Command '$command_to_run' exited with status $cmd_status." >> ecbs_result.txt
    elif [ $cmd_status -eq 0 ]; then
        echo -e "\033[0;32mCommand '$command_to_run' successfully completed.\033[0m"
        echo "Command '$command_to_run' successfully completed." >> ecbs_result.txt
    fi
}

# 여기에 세빈이형 테스트 내용 넣으면 돼요
ENVIRONMENTS=("RN")
ROBOTNUM=(20 25 30)
MAXCOUNT=50

BASE_COMMAND="python centralized/ecbs-ta/ecbs-ta.py"

# 여기는 건들지 말아주세요
for env in "${ENVIRONMENTS[@]}"; do
    for var1 in "${ROBOTNUM[@]}"; do
        for var2 in $(seq 1 $MAXCOUNT); do
            for var3 in 0 1; do
              execute_with_timeout "$BASE_COMMAND $env $var1 $var2 $var3"
            done
        done
    done
done
