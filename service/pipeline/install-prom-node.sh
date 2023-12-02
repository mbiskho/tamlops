#!/bin/bash

# Install & add user dir before install
apt update -y
apt-get install -y wget tar systemctl
adduser --quiet --disabled-password --shell /bin/bash --home /home/muhammad_haqqi01 --gecos "User" muhammad_haqqi01

# Install prometheus
groupadd --system prometheus
useradd -s /sbin/nologin --system -g prometheus prometheus

mkdir /etc/prometheus
mkdir /var/lib/prometheus

cd /home/muhammad_haqqi01
wget https://github.com/prometheus/prometheus/releases/download/v2.43.0/prometheus-2.43.0.linux-amd64.tar.gz
tar vxf prometheus*.tar.gz
cd prometheus*/

mv prometheus /usr/local/bin
mv promtool /usr/local/bin
chown prometheus:prometheus /usr/local/bin/prometheus
chown prometheus:prometheus /usr/local/bin/promtool

mv consoles /etc/prometheus
mv console_libraries /etc/prometheus
mv prometheus.yml /etc/prometheus

chown prometheus:prometheus /etc/prometheus
chown -R prometheus:prometheus /etc/prometheus/consoles
chown -R prometheus:prometheus /etc/prometheus/console_libraries
chown -R prometheus:prometheus /var/lib/prometheus

touch /etc/systemd/system/prometheus.service 
echo "[Unit]
Description=Prometheus
Wants=network-online.target
After=network-online.target

[Service]
User=prometheus
Group=prometheus
Type=simple
ExecStart=/usr/local/bin/prometheus \
    --config.file /etc/prometheus/prometheus.yml \
    --storage.tsdb.path /var/lib/prometheus/ \
    --web.console.templates=/etc/prometheus/consoles \
    --web.console.libraries=/etc/prometheus/console_libraries

[Install]
WantedBy=multi-user.target" >> /etc/systemd/system/prometheus.service

# Install node-exporter
useradd -rs /bin/false node_exporter

cd /home/muhammad_haqqi01
wget https://github.com/prometheus/node_exporter/releases/download/v0.18.1/node_exporter-0.18.1.linux-amd64.tar.gz
tar -xvf node_exporter-0.18.1.linux-amd64.tar.gz
mv node_exporter-0.18.1.linux-amd64/node_exporter /usr/local/bin/

touch /etc/systemd/system/node_exporter.service
echo "[Unit]
Description=Node Exporter
After=network.target

[Service]
User=node_exporter
Group=node_exporter
Type=simple
ExecStart=/usr/local/bin/node_exporter

[Install]
WantedBy=multi-user.target" >> /etc/systemd/system/node_exporter.service

# Reload daemon & run prometheus + node-exporter
systemctl daemon-reload
systemctl enable prometheus
systemctl start prometheus
systemctl enable node_exporter
systemctl start node_exporter

# Check if prometheus + node-exporter is active & running
systemctl status prometheus
systemctl status node_exporter