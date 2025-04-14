#!/bin/bash

function usage {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start     - Start the main services"
    echo "  stop      - Stop the main services"
    echo "  restart   - Restart the main services"
    echo "  status    - Show the status of all services"
    echo "  logs      - Show logs from services"
    echo "  cleanup   - Remove unused Docker resources"
    echo "  prune     - Remove all stopped containers and unused volumes/networks"
    echo ""
    exit 1
}

function start {
    echo "Starting main services..."
    docker-compose up -d
}

function stop {
    echo "Stopping main services..."
    docker-compose down
}

function restart {
    stop
    start
}

function status {
    echo "Main services:"
    docker-compose ps
    
    echo ""
    echo "User instances:"
    docker ps --filter "name=prpl-app-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

function logs {
    if [ -z "$2" ]; then
        docker-compose logs
    else
        docker-compose logs "$2"
    fi
}

function cleanup {
    echo "Cleaning up Docker resources..."
    docker system prune -f
}

function prune {
    echo "Warning: This will remove all stopped containers and unused volumes/networks."
    read -p "Are you sure you want to continue? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker system prune -a --volumes -f
    fi
}

# Check command
if [ $# -eq 0 ]; then
    usage
fi

# Process command
case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        status
        ;;
    logs)
        logs "$@"
        ;;
    cleanup)
        cleanup
        ;;
    prune)
        prune
        ;;
    *)
        usage
        ;;
esac
