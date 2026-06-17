#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy.sh — Instala Docker y levanta el backend LSEC en Debian sin GUI
#
# Uso:
#   chmod +x deploy.sh
#   ./deploy.sh              # primera vez: instala Docker + levanta servicio
#   ./deploy.sh restart      # reiniciar contenedor
#   ./deploy.sh logs         # ver logs en vivo
#   ./deploy.sh stop         # detener
#   ./deploy.sh rebuild      # reconstruir imagen (después de cambios en código)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

COMPOSE_CMD=""

# ── Detectar docker compose ────────────────────────────────────────────────────
_find_compose() {
    if docker compose version &>/dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    elif command -v docker-compose &>/dev/null; then
        COMPOSE_CMD="docker-compose"
    fi
}

# ── Instalar Docker Engine en Debian ──────────────────────────────────────────
install_docker() {
    echo "==> Instalando Docker Engine..."
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg lsb-release

    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/debian/gpg \
        | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
      https://download.docker.com/linux/debian \
      $(lsb_release -cs) stable" \
      | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update -qq
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
        docker-buildx-plugin docker-compose-plugin

    # Permitir usar docker sin sudo (requiere nuevo login)
    sudo usermod -aG docker "$USER"
    echo ""
    echo "  IMPORTANTE: Cierra la sesión y vuelve a entrar para usar Docker"
    echo "  sin sudo. Luego ejecuta ./deploy.sh de nuevo."
    echo ""
}

# ── Subcomandos ───────────────────────────────────────────────────────────────
CMD="${1:-up}"

case "$CMD" in
    install)
        install_docker
        exit 0
        ;;
    up|start)
        _find_compose
        if [ -z "$COMPOSE_CMD" ]; then
            echo "Docker Compose no encontrado. Ejecuta: ./deploy.sh install"
            exit 1
        fi
        echo "==> Construyendo imagen (primera vez tarda ~5-10 min)..."
        $COMPOSE_CMD up -d --build
        echo ""
        echo "✓ Backend iniciado en http://0.0.0.0:8000"
        echo "  Health check: curl http://localhost:8000/health"
        IP=$(hostname -I | awk '{print $1}')
        echo "  Desde la app Android (misma WiFi): http://$IP:8000/"
        ;;
    rebuild)
        _find_compose
        $COMPOSE_CMD down
        $COMPOSE_CMD build --no-cache
        $COMPOSE_CMD up -d
        echo "✓ Imagen reconstruida y contenedor reiniciado."
        ;;
    restart)
        _find_compose
        $COMPOSE_CMD restart
        echo "✓ Contenedor reiniciado."
        ;;
    stop)
        _find_compose
        $COMPOSE_CMD down
        echo "✓ Contenedor detenido."
        ;;
    logs)
        _find_compose
        $COMPOSE_CMD logs -f
        ;;
    status)
        _find_compose
        $COMPOSE_CMD ps
        ;;
    ip)
        IP=$(hostname -I | awk '{print $1}')
        echo "IP local de este servidor: $IP"
        echo "URL para la app Android:   http://$IP:8000/"
        ;;
    *)
        echo "Uso: $0 {install|up|rebuild|restart|stop|logs|status|ip}"
        exit 1
        ;;
esac
