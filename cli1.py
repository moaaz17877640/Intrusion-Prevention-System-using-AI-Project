import pyshark
from datetime import datetime, timedelta
import collections
import pyfiglet
import os
import click
import numpy as np
import joblib
import json
import pandas as pd
from tabulate import tabulate # type: ignore
from pathlib import Path
import ctypes
import sys
from scapy.all import rdpcap

# ========== Admin Check ==========
def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

if not is_admin():
    print("\033[1;31mPlease run the program as administrator!\033[0m")
    sys.exit(1)

# ========== Configuration ==========
INTERFACE = "Wi-Fi 2"
MODEL_DIR = "ml_models"
METADATA_FILE = os.path.join(MODEL_DIR, "model_metadata.json")
BLOCKLIST_FILE = os.path.join(MODEL_DIR, "blocklist.txt")
TEMP_BLOCKLIST_FILE = os.path.join(MODEL_DIR, "temp_blocklist.txt")
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

# Load or define default model metadata
default_models = {}
if os.path.exists(METADATA_FILE):
    try:
        with open(METADATA_FILE, 'r') as f:
            default_models = json.load(f)
    except Exception as e:
        print(f"\033[1;31mError loading metadata: {e}\033[0m")
if not default_models:
    # Default metadata with feature thresholds
    default_models = {
        "default_model": {
            "accuracy": 0.95,
            "threshold": 0.5,
            "feature_order": [
                "flow_rate", "syn_rate", "avg_packet_size", "syn_ratio", "ack_ratio",
                "psh_ratio", "urg_ratio", "entropy", "flow_duration", "down_up_ratio"
            ],
            "feature_thresholds": {
                "flow_rate": 100.0,
                "syn_rate": 10.0,
                "avg_packet_size": 1500.0,
                "syn_ratio": 0.5,
                "ack_ratio": 0.5,
                "psh_ratio": 0.5,
                "urg_ratio": 0.5,
                "entropy": 8.0,
                "flow_duration": 60.0,
                "down_up_ratio": 10.0
            }
        }
    }

# ========== Global Variables ==========
packet_number = 0
flow_counts = collections.defaultdict(int)
syn_counts = collections.defaultdict(int)
psh_counts = collections.defaultdict(int)
ack_counts = collections.defaultdict(int)
urg_counts = collections.defaultdict(int)
packet_sizes = collections.defaultdict(list)
down_up_ratios = collections.defaultdict(float)
flow_start_times = collections.defaultdict(datetime.now)
packet_intervals = collections.defaultdict(list)
flow_durations = collections.defaultdict(float)
suspicious_flows = set()
feature_thresholds = {}  # Dictionary to store feature thresholds

current_model = None
current_scaler = None
model_accuracy = 0.0
feature_order = []
model_threshold = 0.5

# ========== Helper Functions ==========
def display_banner():
    banner = pyfiglet.figlet_format("AI DDoS Detector", font="slant")
    click.echo(f"\033[1;32m{banner}\033[0m")
    click.echo("=" * 80)
    click.echo("Real-time DDoS Detection System")
    click.echo("=" * 80)

def parse_value(value):
    return 1 if str(value).strip().lower() == 'true' else 0

def calculate_entropy(flow_key):
    if not packet_sizes[flow_key]:
        return 0
    counts = np.bincount(packet_sizes[flow_key])
    probs = counts / np.sum(counts)
    return -np.sum([p * np.log2(p) for p in probs if p > 0])

def extract_features(flow_key, current_time):
    try:
        flow_duration = max(flow_durations[flow_key], 0.001)
        flow_count = max(flow_counts[flow_key], 1)
        features = [
            flow_counts[flow_key] / flow_duration,  # flow_rate
            syn_counts[flow_key] / flow_duration,   # syn_rate
            np.mean(packet_sizes[flow_key]) if packet_sizes[flow_key] else 0,  # avg_packet_size
            syn_counts[flow_key] / flow_count,      # syn_ratio
            ack_counts[flow_key] / flow_count,      # ack_ratio
            psh_counts[flow_key] / flow_count,      # psh_ratio
            urg_counts[flow_key] / flow_count,      # urg_ratio
            calculate_entropy(flow_key),            # entropy
            flow_durations[flow_key],               # flow_duration
            down_up_ratios[flow_key]                # down_up_ratio
        ]
        # Replace invalid values
        features = [float(f) if f is not None and not np.isnan(f) else 0.0 for f in features]
        return features
    except Exception as e:
        print(f"\033[1;31mError extracting features: {e}\033[0m")
        return [0.0] * 10  # Return default zeroed features

def ml_based_detection(features):
    global feature_thresholds, feature_order
    if not feature_order:
        print("\033[1;31mError: feature_order is empty. Cannot perform detection.\033[0m")
        return False, 0.0, 0

    if len(features) != len(feature_order):
        print(f"\033[1;31mError: Feature count mismatch. Expected {len(feature_order)}, got {len(features)}.\033[0m")
        return False, 0.0, 0

    # Threshold-based comparison
    exceed_count = 0
    print("\033[1;36mThreshold Comparisons:\033[0m")  # Debug output
    for feature_value, feature_name in zip(features, feature_order):
        threshold = feature_thresholds.get(feature_name)
        if threshold is None:
            print(f"\033[1;33mWarning: No threshold for {feature_name}. Skipping.\033[0m")
            continue
        try:
            threshold = float(threshold)
            print(f"  {feature_name}: {feature_value:.2f} vs {threshold:.2f}")  # Debug
            if feature_value > threshold:
                exceed_count += 1
        except (ValueError, TypeError) as e:
            print(f"\033[1;31mError comparing {feature_name}: {e}\033[0m")

    # ML model prediction
    proba = 0.0
    if current_model and current_scaler:
        try:
            features_df = pd.DataFrame([features], columns=feature_order)
            features_df = features_df.replace([np.inf, -np.inf], 0).fillna(0)
            scaled = current_scaler.transform(features_df)
            proba = current_model.predict_proba(scaled)[0][1]
            print(f"\033[1;36mML Probability: {proba:.2f}\033[0m")  # Debug
        except Exception as e:
            print(f"\033[1;31mML Detection error: {e}\033[0m")
            proba = 0.0
    else:
        print("\033[1;31mError: Model or scaler not loaded. Using threshold-based detection only.\033[0m")

    # Verdict
    if  proba+2 >= model_threshold:
        print(f"\033[1;32mAttack detected: Exceed Count={exceed_count}, Proba={proba:.2f}\033[0m")
        return True, proba * 100, exceed_count
    else:
        print(f"\033[1;33mNo attack: Exceed Count={exceed_count}, Proba={proba:.2f}\033[0m")
        return False, proba * 100, exceed_count

def packet_callback(packet):
    global packet_number
    packet_number += 1
    try:
        if hasattr(packet, 'tcp'):
            flow_key = f"{packet.ip.src}:{packet.tcp.dstport}"
        elif hasattr(packet, 'udp'):
            flow_key = f"{packet.ip.src}:{packet.udp.dstport}"
        else:
            print(f"\033[1;33mSkipping packet: No TCP or UDP layer.\033[0m")
            return

        current_time = datetime.now()
        flow_counts[flow_key] += 1
        packet_sizes[flow_key].append(int(packet.length))
        if hasattr(packet, 'tcp'):
            syn_counts[flow_key] += parse_value(packet.tcp.flags_syn)
            ack_counts[flow_key] += parse_value(packet.tcp.flags_ack)
            psh_counts[flow_key] += parse_value(packet.tcp.flags_push)
            urg_counts[flow_key] += parse_value(packet.tcp.flags_urg)
        
        features = extract_features(flow_key, current_time)
        print(f"\033[1;36mExtracted features: {features}\033[0m")  # Debug output

        ml_result, ml_confidence, exceed_count = ml_based_detection(features)

        # Display feature values and thresholds
        print(f"\nFlow: {flow_key}")
        headers = ['Feature', 'Value', 'Threshold', 'Exceeds']
        data = [
            (name, f"{value:.2f}", f"{feature_thresholds.get(name, 'N/A')}", 
             "Yes" if feature_thresholds.get(name) is not None and value > float(feature_thresholds.get(name)) else "No")
            for name, value in zip(feature_order, features)
        ]
        print(tabulate(data, headers=headers, tablefmt="fancy_grid", floatfmt=".2f"))
        print(f"\n  ML Verdict: {'ATTACK ✅' if ml_result else 'Normal ❌'} "
              f"(Confidence: {ml_confidence:.2f}%, Exceeding Features: {exceed_count})")

        if ml_result:
            suspicious_flows.add(flow_key)
    except Exception as e:
        print(f"\033[1;31mError processing packet: {e}\033[0m")

def load_model_and_show_values(model_path):
    global current_model, current_scaler, model_accuracy, feature_order, model_threshold, feature_thresholds
    try:
        # Load the model
        current_model = joblib.load(model_path)
        scaler_path = model_path.replace('.pkl', '_scaler.pkl')
        if os.path.exists(scaler_path):
            current_scaler = joblib.load(scaler_path)
        else:
            print(f"\033[1;33mWarning: Scaler file '{scaler_path}' not found.\033[0m")
            current_scaler = None

        # Load feature order and thresholds fltrom metadata
        model_name = os.path.basename(model_path).replace('.pkl', '')
        if model_name in default_models:
            metadata = default_models[model_name]
            feature_order = metadata.get("feature_order", metadata.get("features", []))
            feature_thresholds = metadata.get("feature_thresholds", {})
            model_accuracy = metadata.get("accuracy", "Unknown")
            model_threshold = metadata.get("threshold", 0.5)
        else:
            print(f"\033[1;33mModel '{model_name}' not found in metadata. Using default model.\033[0m")
            metadata = default_models["default_model"]
            feature_order = metadata.get("feature_order", [])
            feature_thresholds = metadata.get("feature_thresholds", {})
            model_accuracy = metadata.get("accuracy", 0.95)
            model_threshold = metadata.get("threshold", 0.5)

        # Validate feature_order and feature_thresholds
        if not feature_order:
            print(f"\033[1;31mError: feature_order is empty. Using default.\033[0m")
            feature_order = default_models["default_model"]["feature_order"]
            feature_thresholds = default_models["default_model"]["feature_thresholds"]
        missing_thresholds = [f for f in feature_order if f not in feature_thresholds]
        if missing_thresholds:
            print(f"\033[1;31mError: Missing thresholds for features: {missing_thresholds}. Using defaults.\033[0m")
            for f in missing_thresholds:
                feature_thresholds[f] = default_models["default_model"]["feature_thresholds"].get(f, 0.0)

        print("\n\033[1;32mModel loaded successfully!\033[0m")
        print("\n\033[1;34mModel Feature Thresholds for Comparison:\033[0m")
        print("=" * 50)
        for feature, threshold in feature_thresholds.items():
            print(f"{feature}: {threshold}")
        print(f"Feature Order: {feature_order}")
        print("=" * 50)

    except Exception as e:
        print(f"\033[1;31mError loading model: {e}\033[0m")
        # Fallback to default
        feature_order = default_models["default_model"]["feature_order"]
        feature_thresholds = default_models["default_model"]["feature_thresholds"]
        model_accuracy = default_models["default_model"]["accuracy"]
        model_threshold = default_models["default_model"]["threshold"]

def analyze_pcap(pcap_file):
    global suspicious_flows
    suspicious_flows.clear()
    print(f"\n\033[1;34mAnalyzing {pcap_file}...\033[0m")
    try:
        capture = pyshark.FileCapture(pcap_file)
        capture.apply_on_packets(packet_callback)

        suspicious_ips = list({flow.split(':')[0] for flow in suspicious_flows})

        print("\nFlow Verdicts:")
        print("=" * 50)
        for flow in set(flow_counts.keys()):
            ip = flow.split(":")[0]
            if flow in suspicious_flows:
                print(f"{ip} -> ATTACK DETECTED ✅")
            else:
                print(f"{ip} -> Normal ❌")
        print("=" * 50)

        if suspicious_ips:
            print("\nDetected IPs:")
            for idx, ip in enumerate(suspicious_ips, 1):
                print(f"{idx}) {ip}")
            print("\n\033[1;36mChoose action: 1) Temp Block | 2) Perm Block | 3) Ignore\033[0m")
            for ip in suspicious_ips:
                choice = input(f"Action for {ip}: ").strip()
                execute_action(ip, choice)
        else:
            print("No malicious activity detected")

    except Exception as e:
        print(f"\033[1;31mAnalysis failed: {e}\033[0m")

# ========== Block Management ==========
def execute_action(ip, choice):
    try:
        if choice == '1':
            expiry = datetime.now() + timedelta(hours=1)
            os.system(
                f"netsh advfirewall firewall add rule name=\"TempBlock_{ip}\" "
                f"dir=in action=block remoteip={ip} enable=yes profile=any"
            )
            with open(TEMP_BLOCKLIST_FILE, 'a') as f:
                f.write(f"{ip}|{expiry}\n")
            print(f"\033[1;32mTemporarily blocked {ip} until {expiry.strftime('%Y-%m-%d %H:%M')}\033[0m")
        elif choice == '2':
            with open(BLOCKLIST_FILE, 'a') as f:
                f.write(f"{ip}\n")
            print(f"\033[1;32mPermanently blocked {ip}\033[0m")
        else:
            print(f"\033[1;33mIgnored {ip}\033[0m")
    except Exception as e:
        print(f"\033[1;31mAction failed: {e}\033[0m")

def show_blocked_ips():
    print("\n\033[1;35mBlocked IPs Report:\033[0m")
    print("="*50)
    if os.path.exists(BLOCKLIST_FILE):
        with open(BLOCKLIST_FILE, 'r') as f:
            perm_ips = [line.strip() for line in f.readlines()]
        print("\nPermanently Blocked:")
        for ip in perm_ips:
            print(f" - {ip}")
    if os.path.exists(TEMP_BLOCKLIST_FILE):
        with open(TEMP_BLOCKLIST_FILE, 'r') as f:
            temp_ips = [line.strip().split('|') for line in f.readlines()]
        print("\nTemporarily Blocked:")
        for ip, expiry in temp_ips:
            print(f" - {ip} (Expires: {expiry})")
    print("="*50)

def unblock_ip(ip):
    try:
        if os.path.exists(BLOCKLIST_FILE):
            with open(BLOCKLIST_FILE, 'r') as f:
                perm_ips = [line.strip() for line in f]
            if ip in perm_ips:
                perm_ips.remove(ip)
                with open(BLOCKLIST_FILE, 'w') as f:
                    f.write("\n".join(perm_ips))
                os.system(f"netsh advfirewall firewall delete rule name=\"PermBlock_{ip}\"")
                print(f"\033[1;32mUnblocked {ip} from permanent list\033[0m")
        if os.path.exists(TEMP_BLOCKLIST_FILE):
            with open(TEMP_BLOCKLIST_FILE, 'r') as f:
                temp_ips = [line.strip().split('|') for line in f]
            remaining = [line for line in temp_ips if line[0] != ip]
            with open(TEMP_BLOCKLIST_FILE, 'w') as f:
                for item in remaining:
                    f.write("|".join(item) + "\n")
            os.system(f"netsh advfirewall firewall delete rule name=\"TempBlock_{ip}\"")
            print(f"\033[1;32mUnblocked {ip} from temporary list\033[0m")
    except Exception as e:
        print(f"\033[1;31mError unblocking: {e}\033[0m")

# ========== Main Menu ==========
def main_menu():
    while True:
        display_banner()
        print("1) Load Model & Analyze PCAP")
        print("2) Show Blocked IPs")
        print("3) Unblock IP")
        print("4) Exit\n")

        choice = input("Select option: ").strip()

        if choice == '1':
            model_path = input("\nEnter full path of the model file: ").strip()
            load_model_and_show_values(model_path)
            analyze_pcap(input("\nEnter PCAP file path: "))
        elif choice == '2':
            show_blocked_ips()
        elif choice == '3':
            ip = input("\nEnter IP to unblock: ").strip()
            unblock_ip(ip)
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid option!")

if __name__ == "__main__":
    main_menu()