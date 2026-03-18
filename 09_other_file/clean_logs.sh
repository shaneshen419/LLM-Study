#!/bin/bash

# =================================================================
# 脚本名称: clean_logs_by_date.sh
# 描述: 根据文件名中的日期字符串，清理3天前的日志，正则表达式进行清理
# crontab -e    在里面添加一行：
# 0 0 * * * /bin/bash /data/llm/wra/shane/wra/nexchip-wra-prototype/test_sh/clean_logs.sh >> /data/llm/wra/shane/wra/nexchip-wra-prototype/logs/cleanup.log 2>&1
# =================================================================

# BASE_DIR="/data/llm/wra_llm_test_server/wra_portal"
BASE_DIR="/data/llm/wra/shane/wra/nexchip-wra-prototype"
LOGS_DIR="${BASE_DIR}/logs"
DUMP_DIR="${BASE_DIR}/logs/logs_dump"

# 获取3天前的日期字符串
# 假设今天是2026-03-18，DATE_THRESHOLD 为 20260315
DATE_THRESHOLD=$(date -d "3 days ago" +%Y%m%d)
# 带有横杠的格式，如 2026-03-15
DATE_THRESHOLD_HYPHEN=$(date -d "3 days ago" +%Y-%m-%d)

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

clean_task() {
    log_message "开始执行日志清理任务..."
    log_message "开始执行日期比对清理任务 (阈值: $DATE_THRESHOLD / $DATE_THRESHOLD_HYPHEN)..."

    # 1. 处理 logs 目录 (匹配类似 query_rca_2026-03-11.log)
    if [ -d "$LOGS_DIR" ]; then
        cd "$LOGS_DIR"
        for file in *.log; do
            # 提取文件名中的日期部分 (匹配 YYYY-MM-DD)
            file_date=$(echo "$file" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}')
            if [[ -n "$file_date" ]]; then
                # 去掉横杠方便比较数字大小
                file_date_num=$(echo "$file_date" | sed 's/-//g')
                if [ "$file_date_num" -lt "$DATE_THRESHOLD" ]; then
                    rm -v "$file"
                fi
            fi
        done
    fi

    # 2. 处理 logs_dump 目录 (匹配类似 rca_6397_..._20260311_110113.json)
    if [ -d "$DUMP_DIR" ]; then
        cd "$DUMP_DIR"
        for file in *.json; do
            # 提取文件名中的日期部分 (匹配 8位数字)
            # 注意：排除掉像 6397 这样的小数字，只提取 2026xxxx 这样的部分
            file_date=$(echo "$file" | grep -oE '20[0-9]{6}')
            if [[ -n "$file_date" ]]; then
                if [ "$file_date" -lt "$DATE_THRESHOLD" ]; then
                    rm -v "$file"
                fi
            fi
        done
    fi

    log_message "清理任务完成。"
}

clean_task

