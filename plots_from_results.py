#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Dimitrios Kafetzis
#
# This file is part of the Transformer Inference Simulator project.
# Licensed under the MIT License; you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#   https://opensource.org/licenses/MIT
#
# Author:  Dimitrios Kafetzis (dimitrioskafetzis@gmail.com)
# File:    plots_from_results.py
# Description:
#   Reads an experiment_results.json (like your toy_comparison scenario output)
#   and generates a series of plots, saving each as .png and .pdf.
# Usage:
#   python plots_from_results.py --input /path/to/experiment_results.json --output /path/to/plots_dir
#
# ---------------------------------------------------------------------------

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Generate plots from experiment_results.json.")
    parser.add_argument("--input", required=True, help="Path to experiment_results.json file")
    parser.add_argument("--output", required=True, help="Directory to store output plots")
    args = parser.parse_args()

    # 1) Load JSON
    with open(args.input, "r") as f:
        results_data = json.load(f)
    
    # Adjust these keys based on your actual JSON structure.
    # For your example: results_data["toy_comparison"]["toy_comparison"]["metrics"]
    scenario_key = "toy_comparison"
    scenario_data = results_data[scenario_key][scenario_key]
    metrics = scenario_data["metrics"]
    
    # "comparison_metrics" is where we find latency, resource_utilization, communication_overhead, etc.
    comparison_metrics = metrics["comparison_metrics"]

    # 2) Prepare output directory
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------
    # Helper function to save a figure as both PNG and PDF
    def save_figure(fig, base_filename: str):
        png_path = out_dir / f"{base_filename}.png"
        pdf_path = out_dir / f"{base_filename}.pdf"
        fig.savefig(png_path, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)

    # --------------------------------------------------------------------
    # 3) Plot: Latency progression per step
    #    => comparison_metrics["latency"] => {algo_name: [list_of_latencies_per_step]}
    latencies_dict = comparison_metrics["latency"]

    fig = plt.figure(figsize=(8,6))
    for algo_name, lat_values in latencies_dict.items():
        steps = list(range(len(lat_values)))
        plt.plot(steps, lat_values, marker='o', label=algo_name)
    
    plt.title("Latency Progression per Step")
    plt.xlabel("Step")
    plt.ylabel("Latency (arbitrary units)")
    plt.legend()
    plt.grid(True)
    save_figure(fig, "latency_per_step")

    # --------------------------------------------------------------------
    # 4) Plot: Average latency bar chart
    #    => scenario_data["metrics"]["summary"]["average_latency"]
    avg_lat_dict = metrics["summary"]["average_latency"]
    algo_names = list(avg_lat_dict.keys())
    avg_lats = [avg_lat_dict[a] for a in algo_names]

    fig = plt.figure(figsize=(8,6))
    x_positions = np.arange(len(algo_names))
    plt.bar(x_positions, avg_lats, color="cornflowerblue")
    plt.xticks(x_positions, algo_names, rotation=20)
    plt.title("Average Latency Comparison")
    plt.ylabel("Average Latency")
    plt.grid(axis="y")
    save_figure(fig, "average_latency")

    # --------------------------------------------------------------------
    # 5) Plot: Communication Overhead per Step
    #    => comparison_metrics["communication_overhead"] => {algo_name: [list_of_comm_times]}
    comm_dict = comparison_metrics["communication_overhead"]

    fig = plt.figure(figsize=(8,6))
    for algo_name, comm_vals in comm_dict.items():
        steps = list(range(len(comm_vals)))
        plt.plot(steps, comm_vals, marker='x', label=algo_name)

    plt.title("Communication Overhead per Step")
    plt.xlabel("Step")
    plt.ylabel("Comm Time (arbitrary units)")
    plt.legend()
    plt.grid(True)
    save_figure(fig, "communication_overhead_per_step")

    # --------------------------------------------------------------------
    # 6) Plot: Resource utilization per step
    #    => comparison_metrics["resource_utilization"] => {algo_name: [list_of_utilizations]}
    util_dict = comparison_metrics["resource_utilization"]

    fig = plt.figure(figsize=(8,6))
    for algo_name, util_vals in util_dict.items():
        steps = list(range(len(util_vals)))
        plt.plot(steps, util_vals, marker='s', label=algo_name)

    plt.title("Average Compute Utilization per Step")
    plt.xlabel("Step")
    plt.ylabel("Compute Utilization (fraction)")
    plt.legend()
    plt.grid(True)
    save_figure(fig, "resource_utilization_per_step")

    # --------------------------------------------------------------------
    # 7) Total data transferred for each algorithm (if present)
    #    => metrics["summary"].get("total_data_transferred_each_algo", {})
    if "total_data_transferred_each_algo" in metrics["summary"]:
        data_tx_summary = metrics["summary"]["total_data_transferred_each_algo"]
        algo_names_dt = list(data_tx_summary.keys())
        data_values = [data_tx_summary[a] for a in algo_names_dt]

        fig = plt.figure(figsize=(8,6))
        x_positions = np.arange(len(algo_names_dt))
        plt.bar(x_positions, data_values, color="salmon")
        plt.xticks(x_positions, algo_names_dt, rotation=20)
        plt.title("Total Data Transferred per Algorithm")
        plt.ylabel("Data Transferred (GB)")
        plt.grid(axis="y")
        save_figure(fig, "total_data_transferred")

    print(f"Plots successfully saved in: {out_dir}")

if __name__ == "__main__":
    main()
