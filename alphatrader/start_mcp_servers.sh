#!/bin/bash
# start_mcp_servers.sh
# --------------------
# Launch all 6 MCP servers in the background.
# Run this from the alpha-trader/ root directory.
#
# Usage:
#   chmod +x start_mcp_servers.sh
#   ./start_mcp_servers.sh
#
# To stop all servers:
#   pkill -f "mcp_servers"

echo "Starting AlphaTrader MCP Servers..."
echo ""

# Activate venv if present
if [ -f "backend/.venv/bin/activate" ]; then
  source backend/.venv/bin/activate
fi

# Start each server in background
python mcp_servers/market_data/server.py  &  echo "  [8001] market_data   PID=$!"
python mcp_servers/news/server.py         &  echo "  [8002] news          PID=$!"
python mcp_servers/financials/server.py   &  echo "  [8003] financials    PID=$!"
python mcp_servers/orders/server.py       &  echo "  [8004] orders        PID=$!"
python mcp_servers/risk/server.py         &  echo "  [8005] risk          PID=$!"
python mcp_servers/notifications/server.py & echo "  [8006] notifications PID=$!"

echo ""
echo "All 6 MCP servers running. Start the backend next:"
echo "  cd backend && uvicorn main:app --reload"
echo ""
echo "To stop all servers: pkill -f 'mcp_servers'"

wait
