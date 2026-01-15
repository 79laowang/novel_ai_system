#!/bin/bash
# 实时监控调试日志

echo "======================================"
echo "监控调试日志"
echo "======================================"
echo ""
echo "日志文件: /home/kewang/work/novel_ai_system/logs/debug.log"
echo ""
echo "按 Ctrl+C 退出"
echo ""
echo "======================================"
echo ""

tail -f /home/kewang/work/novel_ai_system/logs/debug.log
