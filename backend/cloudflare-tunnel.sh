#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# cloudflare-tunnel.sh — Expone el backend LSEC a internet de forma segura
#
# Usa Cloudflare Tunnel (cloudflared):
#   - HTTPS automático con certificado de Cloudflare
#   - Sin abrir puertos en el router
#   - Protección DDoS incluida gratis
#   - Funciona sin dominio propio (opción quick-tunnel)
#
# Uso:
#   chmod +x cloudflare-tunnel.sh
#   ./cloudflare-tunnel.sh quick     # URL temporal (sin cuenta, reinicia al apagar)
#   ./cloudflare-tunnel.sh install   # Instalar cloudflared como servicio permanente
#   ./cloudflare-tunnel.sh status    # Ver estado del túnel
#   ./cloudflare-tunnel.sh stop      # Detener el servicio
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

TUNNEL_NAME="lsec-backend"
LOCAL_URL="http://localhost:8000"

# ── Instalar cloudflared ───────────────────────────────────────────────────────
install_cloudflared() {
    if command -v cloudflared &>/dev/null; then
        echo "cloudflared ya está instalado: $(cloudflared --version)"
        return
    fi

    echo "==> Instalando cloudflared..."
    ARCH=$(dpkg --print-architecture)  # amd64 o arm64

    curl -fsSL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${ARCH}.deb" \
        -o /tmp/cloudflared.deb
    sudo dpkg -i /tmp/cloudflared.deb
    rm /tmp/cloudflared.deb
    echo "cloudflared instalado: $(cloudflared --version)"
}

# ── Opción 1: Quick Tunnel (sin cuenta, URL aleatoria) ────────────────────────
# La URL cambia cada vez que se inicia. Útil para pruebas puntuales.
quick_tunnel() {
    install_cloudflared
    echo ""
    echo "==> Iniciando Quick Tunnel (URL temporal)..."
    echo "    La URL aparece en la siguiente línea — cópiala a la app Android."
    echo "    Presiona Ctrl+C para detener."
    echo ""
    cloudflared tunnel --url "$LOCAL_URL"
}

# ── Opción 2: Túnel permanente con cuenta Cloudflare ──────────────────────────
# La URL es fija (ej: lsec-backend.tu-dominio.com) y sobrevive reinicios.
# Requiere: cuenta gratuita en cloudflare.com + dominio (o subdominio gratuito).
setup_permanent() {
    install_cloudflared

    echo ""
    echo "==> Configurando túnel permanente..."
    echo ""
    echo "Paso 1: Autenticar con Cloudflare (abrirá un enlace — cópialo al navegador)"
    cloudflared tunnel login

    echo ""
    echo "Paso 2: Crear el túnel '$TUNNEL_NAME'"
    cloudflared tunnel create "$TUNNEL_NAME"

    TUNNEL_ID=$(cloudflared tunnel list | grep "$TUNNEL_NAME" | awk '{print $1}')
    CONFIG_DIR="$HOME/.cloudflared"
    CONFIG_FILE="$CONFIG_DIR/config.yml"

    echo ""
    echo "Paso 3: Generando configuración en $CONFIG_FILE"
    cat > "$CONFIG_FILE" <<EOF
tunnel: $TUNNEL_ID
credentials-file: $CONFIG_DIR/$TUNNEL_ID.json

ingress:
  - hostname: lsec.${TUNNEL_NAME}.cfargotunnel.com
    service: $LOCAL_URL
  - service: http_status:404
EOF

    echo "Paso 4: Instalar como servicio systemd (arranca automáticamente)"
    sudo cloudflared service install
    sudo systemctl enable cloudflared
    sudo systemctl start cloudflared

    echo ""
    echo "=== TUNEL LISTO ==="
    echo "URL pública: https://lsec.${TUNNEL_NAME}.cfargotunnel.com"
    echo "Usa esta URL en la app Android -> Configuración -> URL del servidor"
    echo ""
    echo "Para ver logs: journalctl -u cloudflared -f"
}

# ── Subcomandos ───────────────────────────────────────────────────────────────
CMD="${1:-quick}"

case "$CMD" in
    quick)
        quick_tunnel
        ;;
    install|permanent)
        setup_permanent
        ;;
    status)
        if systemctl is-active --quiet cloudflared 2>/dev/null; then
            echo "cloudflared: ACTIVO"
            systemctl status cloudflared --no-pager
        else
            echo "cloudflared: no está corriendo como servicio"
            echo "Ejecuta: ./cloudflare-tunnel.sh quick  (para prueba rápida)"
            echo "     o:  ./cloudflare-tunnel.sh install (para servicio permanente)"
        fi
        ;;
    stop)
        sudo systemctl stop cloudflared
        echo "cloudflared detenido."
        ;;
    logs)
        journalctl -u cloudflared -f
        ;;
    *)
        echo "Uso: $0 {quick|install|status|stop|logs}"
        exit 1
        ;;
esac
