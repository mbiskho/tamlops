provider "google" {
  credentials = file("../keys.json")
  project     = "mlops-398205"
}

resource "google_compute_instance" "default" {
  name         = "interface-proxy"
  machine_type = "n1-standard-2" 
  zone         = "us-central1-a"

  boot_disk {
    initialize_params {
      image = "ubuntu-2004-lts"
    }
  }

  network_interface {
    network = "default"
    access_config {}
  }
}

# Define a firewall rule to allow incoming SSH (port 22) and HTTP (port 80) traffic
resource "google_compute_firewall" "allow_ssh_http" {
  name    = "allow-ssh-http"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["22", "80"]
  }

  source_ranges = ["0.0.0.0/0"]
}


# Retrieve the external IP address using a data source
data "google_compute_instance" "default" {
  name         = google_compute_instance.default.name
  zone         = google_compute_instance.default.zone
}

output "instance_ips" {
  value = "${join(" ", google_compute_instance.default.*.network_interface.0.access_config.0.nat_ip)}"
  description = "The public IP address of the newly created instance"
}