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
    scenario_key = "toy_comparison"
    if scenario_key not in results_data or not isinstance(results_data[scenario_key], dict):
        print(f"Error: '{scenario_key}' not found or not a dict in the top-level results.")
        return

    subscenario_data = results_data[scenario_key]
    if scenario_key not in subscenario_data or not isinstance(subscenario_data[scenario_key], dict):
        print(f"Error: '{scenario_key}' not found or not a dict under results_data['{scenario_key}'].")
        return

    scenario_data = subscenario_data[scenario_key]
    if "metrics" not in scenario_data or not isinstance(scenario_data["metrics"], dict):
        print("Error: 'metrics' not found or not a dict in the scenario data.")
        return

    metrics = scenario_data["metrics"]
    
    # "comparison_metrics" is where we find latency, resource_utilization, etc.
    if "comparison_metrics" not in metrics or not isinstance(metrics["comparison_metrics"], dict):
        print("Error: 'comparison_metrics' not found or not a dict in metrics.")
        return

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
    if "latency" in comparison_metrics and isinstance(comparison_metrics["latency"], dict):
        latencies_dict = comparison_metrics["latency"]

        fig = plt.figure(figsize=(8,6))
        for algo_name, lat_values in latencies_dict.items():
            if not isinstance(lat_values, list):
                print(f"Warning: 'latency' data for '{algo_name}' is not a list; skipping.")
                continue
            steps = list(range(len(lat_values)))
            plt.plot(steps, lat_values, marker='o', markersize=2, label=algo_name)
        
        plt.title("Latency Progression per Step")
        plt.xlabel("Step")
        plt.ylabel("Latency (arbitrary units)")
        plt.legend()
        plt.grid(True)
        save_figure(fig, "latency_per_step")
    else:
        print("Warning: 'latency' dictionary not found in comparison_metrics; skipping latency plot.")

    # --------------------------------------------------------------------
    # 4) Plot: Average latency bar chart
    summary_data = metrics.get("summary", {})
    avg_lat_dict = summary_data.get("average_latency", None)
    if isinstance(avg_lat_dict, dict):
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
    else:
        print("Warning: 'average_latency' is missing or not a dict; skipping average latency bar chart.")

    # --------------------------------------------------------------------
    # 5) Plot: Communication Overhead per Step
    if "communication_overhead" in comparison_metrics and isinstance(comparison_metrics["communication_overhead"], dict):
        comm_dict = comparison_metrics["communication_overhead"]

        fig = plt.figure(figsize=(8,6))
        for algo_name, comm_vals in comm_dict.items():
            if not isinstance(comm_vals, list):
                print(f"Warning: 'communication_overhead' data for '{algo_name}' is not a list; skipping.")
                continue
            steps = list(range(len(comm_vals)))
            plt.plot(steps, comm_vals, marker='x', markersize=2, label=algo_name)

        plt.title("Communication Overhead per Step")
        plt.xlabel("Step")
        plt.ylabel("Comm Time (arbitrary units)")
        plt.legend()
        plt.grid(True)
        save_figure(fig, "communication_overhead_per_step")
    else:
        print("Warning: 'communication_overhead' dictionary not found in comparison_metrics; skipping that plot.")

    # --------------------------------------------------------------------
    # 6) Plot: Average Compute Utilization per Step
    if "resource_utilization" in comparison_metrics and isinstance(comparison_metrics["resource_utilization"], dict):
        util_dict = comparison_metrics["resource_utilization"]

        fig = plt.figure(figsize=(8,6))
        for algo_name, util_vals in util_dict.items():
            if not isinstance(util_vals, list):
                print(f"Warning: 'resource_utilization' data for '{algo_name}' is not a list; skipping.")
                continue
            steps = list(range(len(util_vals)))
            plt.plot(steps, util_vals, marker='s', markersize=2, label=algo_name)

        plt.title("Average Compute Utilization per Step")
        plt.xlabel("Step")
        plt.ylabel("Compute Utilization (fraction)")
        plt.legend()
        plt.grid(True)
        save_figure(fig, "resource_utilization_per_step")
    else:
        print("Warning: 'resource_utilization' dictionary missing; skipping compute utilization plot.")

    # --------------------------------------------------------------------
    # 7) Total data transferred for each algorithm (if present)
    data_tx_summary = summary_data.get("total_data_transferred_each_algo", None)
    if isinstance(data_tx_summary, dict):
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
    else:
        print("Info: 'total_data_transferred_each_algo' not found or not dict; skipping data transferred plot.")

    # --------------------------------------------------------------------
    # 8) Feasibility Over Steps (1=feasible, 0=infeasible)
    performance_metrics = scenario_data["metrics"].get("performance_metrics", {})
    if isinstance(performance_metrics, dict):
        step_keys = [k for k in performance_metrics.keys() if k.isdigit()]
        step_keys.sort(key=lambda x: int(x))  # numeric sort

        latencies_dict = comparison_metrics.get("latency", {})
        if not isinstance(latencies_dict, dict):
            latencies_dict = {}

        all_algos = list(latencies_dict.keys())
        feasibility_dict = {algo_name: [] for algo_name in all_algos}

        for step_key in step_keys:
            algo_map = performance_metrics[step_key]
            for algo_name in all_algos:
                lat_val = algo_map.get(algo_name, {}).get("latency", float('inf'))
                if isinstance(lat_val, (float, int)):
                    feasible = 0 if np.isinf(lat_val) else 1
                else:
                    feasible = 0 if str(lat_val).lower() == "infinity" else 1
                feasibility_dict[algo_name].append(feasible)

        if all_algos:
            fig = plt.figure(figsize=(8,6))
            for algo_name, feas_list in feasibility_dict.items():
                steps = list(range(len(feas_list)))
                plt.plot(steps, feas_list, marker='o', markersize=2, label=algo_name)
            plt.title("Feasibility Over Steps")
            plt.xlabel("Step")
            plt.ylabel("Feasible (1) / Infeasible (0)")
            plt.yticks([0,1], ["Infeasible","Feasible"])
            plt.legend()
            plt.grid(True)
            save_figure(fig, "feasibility_over_steps")
        else:
            print("Warning: no valid algorithms found for feasibility plot; skipping.")
    else:
        print("Warning: 'performance_metrics' not found or not dict; skipping feasibility plot.")

    # --------------------------------------------------------------------
    # 9) Consolidated Boxplot for all steps: device compute utilization
    resource_metrics = scenario_data["metrics"].get("resource_metrics", {})
    if isinstance(resource_metrics, dict) and resource_metrics:
        step_keys_resource = sorted(resource_metrics.keys(), key=lambda x: int(x) if x.isdigit() else 999999)
        algo_to_all_utils = {}
        all_algos_in_resource = set()

        for s_key in step_keys_resource:
            step_data = resource_metrics[s_key]
            if isinstance(step_data, dict):
                for algo in step_data.keys():
                    all_algos_in_resource.add(algo)

        all_algos_sorted = sorted(list(all_algos_in_resource))
        for algo in all_algos_sorted:
            algo_to_all_utils[algo] = []

        for s_key in step_keys_resource:
            step_data = resource_metrics[s_key]
            if not isinstance(step_data, dict):
                continue
            for A in step_data.keys():
                dev_map = step_data[A]
                if isinstance(dev_map, dict):
                    for d_id in dev_map:
                        c_util = dev_map[d_id].get("compute_utilization", np.nan)
                        algo_to_all_utils[A].append(c_util)

        fig = plt.figure(figsize=(8,6))
        data_for_boxplot = []
        for A in all_algos_sorted:
            data_for_boxplot.append(algo_to_all_utils[A])

        if data_for_boxplot:
            plt.boxplot(data_for_boxplot, labels=all_algos_sorted)
            plt.title("Compute Utilization Distribution (All Steps)")
            plt.ylabel("Compute Utilization")
            plt.grid(axis="y")
            save_figure(fig, "boxplot_all_steps_compute_util")
        else:
            print("Info: No valid compute utilization data to boxplot.")
    else:
        print("Warning: 'resource_metrics' not found or empty; skipping compute utilization boxplot.")

    # --------------------------------------------------------------------
    # 10) Memory utilization progression over steps for each algo
    if isinstance(resource_metrics, dict) and resource_metrics:
        step_keys_resource = sorted(resource_metrics.keys(), key=lambda x: int(x) if x.isdigit() else 999999)
        memory_util_dict = {}

        all_algos_in_resource = set()
        for s_key in step_keys_resource:
            step_data = resource_metrics[s_key]
            if isinstance(step_data, dict):
                for algo_name in step_data.keys():
                    all_algos_in_resource.add(algo_name)

        for algo_name in all_algos_in_resource:
            memory_util_dict[algo_name] = []

        for s_key in step_keys_resource:
            step_data = resource_metrics[s_key]
            if not isinstance(step_data, dict):
                for a in all_algos_in_resource:
                    memory_util_dict[a].append(np.nan)
                continue

            for algo_name in all_algos_in_resource:
                if algo_name not in step_data:
                    memory_util_dict[algo_name].append(np.nan)
                    continue
                dev_map = step_data[algo_name]
                if not isinstance(dev_map, dict) or not dev_map:
                    memory_util_dict[algo_name].append(np.nan)
                    continue
                sum_mem = 0.0
                cnt = 0
                for d_id in dev_map:
                    mu = dev_map[d_id].get("memory_utilization", np.nan)
                    if not np.isnan(mu):
                        sum_mem += mu
                        cnt += 1
                avg_mem_util = sum_mem / cnt if cnt > 0 else np.nan
                memory_util_dict[algo_name].append(avg_mem_util)

        if memory_util_dict:
            fig = plt.figure(figsize=(8,6))
            for algo_name in sorted(memory_util_dict.keys()):
                mem_vals = memory_util_dict[algo_name]
                steps = list(range(len(mem_vals)))
                plt.plot(steps, mem_vals, marker='o', markersize=2, label=algo_name)
            plt.title("Average Memory Utilization per Step")
            plt.xlabel("Step")
            plt.ylabel("Memory Utilization (fraction)")
            plt.legend()
            plt.grid(True)
            save_figure(fig, "memory_utilization_per_step")
        else:
            print("Info: memory utilization dictionary is empty; skipping.")
    else:
        print("Warning: no resource_metrics for memory utilization progression; skipping.")

    # --------------------------------------------------------------------
    # 11) Latency vs. Communication Overhead scatter, per step
    lat_dict = comparison_metrics.get("latency", {})
    comm_overhead_dict = comparison_metrics.get("communication_overhead", {})
    if isinstance(lat_dict, dict) and isinstance(comm_overhead_dict, dict):
        fig = plt.figure(figsize=(8,6))
        for algo_name in lat_dict:
            lat_vals = lat_dict[algo_name]
            comm_vals = comm_overhead_dict.get(algo_name, [])
            if (not isinstance(lat_vals, list)) or (not isinstance(comm_vals, list)):
                print(f"Warning: For '{algo_name}', lat_vals or comm_vals are not lists. Skipping scatter.")
                continue
            x = []
            y = []
            for lv, cv in zip(lat_vals, comm_vals):
                if isinstance(lv, (float,int)) and not np.isinf(lv):
                    x.append(lv)
                    y.append(cv)
            plt.scatter(x, y, label=algo_name, alpha=0.7, s=20)
        plt.title("Latency vs Communication Overhead (All Steps)")
        plt.xlabel("Latency")
        plt.ylabel("Comm Time")
        plt.legend()
        plt.grid(True)
        save_figure(fig, "scatter_latency_vs_comm")
    else:
        print("Info: 'latency' or 'communication_overhead' missing in comparison_metrics; skipping scatter plot.")

    # --------------------------------------------------------------------
    # 12) Cumulative Summation of Latency Over Steps
    if isinstance(lat_dict, dict):
        fig = plt.figure(figsize=(8,6))
        plotted_any = False
        for algo_name, lat_list in lat_dict.items():
            if not isinstance(lat_list, list):
                print(f"Warning: Latency data for '{algo_name}' is not a list, skipping cumulative sum.")
                continue
            finite_vals = []
            for lv in lat_list:
                if isinstance(lv, (float,int)) and not np.isinf(lv):
                    finite_vals.append(lv)
                else:
                    finite_vals.append(999999)
            if finite_vals:
                cumsum_vals = np.cumsum(finite_vals)
                steps = list(range(len(finite_vals)))
                plt.plot(steps, cumsum_vals, marker='o', markersize=2, label=algo_name)
                plotted_any = True
        if plotted_any:
            plt.title("Cumulative Latency Over Steps")
            plt.xlabel("Step")
            plt.ylabel("Cumulative Latency")
            plt.legend()
            plt.grid(True)
            save_figure(fig, "cumulative_latency")
        else:
            plt.close(fig)
            print("Info: No valid latency data for cumsum plot; skipping.")
    else:
        print("Info: 'latency' is not a dict; skipping cumulative latency.")

    # --------------------------------------------------------------------
    # 13) Migrations
    migrations_map = comparison_metrics.get("migration_counts", {})
    if isinstance(migrations_map, dict) and migrations_map:
        # If it's just an int per algo => bar, else line
        all_int = all(isinstance(migrations_map[a], int) for a in migrations_map)
        if all_int:
            fig = plt.figure(figsize=(8,6))
            algo_list = list(migrations_map.keys())
            migr_counts = [migrations_map[a] for a in algo_list]
            x_pos = np.arange(len(algo_list))
            plt.bar(x_pos, migr_counts, color="gray")
            plt.xticks(x_pos, algo_list, rotation=20)
            plt.title("Total Migrations by Algorithm")
            plt.ylabel("Migrations (count)")
            plt.grid(axis="y")
            save_figure(fig, "migrations_bar")
        else:
            # Possibly a list of per-step migration counts
            fig = plt.figure(figsize=(8,6))
            plotted_any_migration = False
            for algo_name, val in migrations_map.items():
                if isinstance(val, list):
                    steps = list(range(len(val)))
                    plt.plot(steps, val, marker='o', markersize=2, label=algo_name)
                    plotted_any_migration = True
            if plotted_any_migration:
                plt.title("Migrations Per Step")
                plt.xlabel("Step")
                plt.ylabel("Migrations")
                plt.grid(True)
                plt.legend()
                save_figure(fig, "migrations_line")
            else:
                plt.close(fig)
                print("Info: No per-step migration data found.")
    else:
        print("Info: 'migration_counts' not found or empty in comparison_metrics; skipping migration plot.")

    # --------------------------------------------------------------------
    # 14) COMPARISON TO OPTIMAL LATENCY (as percentage difference)
    #
    # We'll do two plots:
    #   A) Step-by-step "percentage difference" from the optimal
    #   B) Bar chart of average percentage difference across steps.
    #
    lat_dict = comparison_metrics.get("latency", {})
    if not isinstance(lat_dict, dict):
        print("Info: 'latency' missing or not a dict, skipping comparison to optimal.")
    elif "exact_optimal" not in lat_dict:
        print("Info: 'exact_optimal' not found in lat_dict; skipping comparison to optimal.")
    else:
        optimal_vals = lat_dict["exact_optimal"]
        if not isinstance(optimal_vals, list):
            print("Warning: 'exact_optimal' latencies are not a list; skipping.")
        else:
            # We'll compute % difference for each step, for each other algo:
            #  difference% = ((base_val - opt_val) / opt_val) * 100
            percent_diff_dict = {}
            all_algos = list(lat_dict.keys())
            baseline_algos = [a for a in all_algos if a != "exact_optimal"]

            # A) Step-by-step percentage difference
            for algo_name in baseline_algos:
                percent_diff_dict[algo_name] = []
                algo_lat_vals = lat_dict[algo_name]
                if not isinstance(algo_lat_vals, list):
                    print(f"Warning: '{algo_name}' latencies are not a list; skipping difference.")
                    continue
                for step_idx, opt_val in enumerate(optimal_vals):
                    base_val = float('inf')
                    if step_idx < len(algo_lat_vals):
                        base_val = algo_lat_vals[step_idx]

                    # If optimal or base is missing or infinite => set to np.nan
                    if (isinstance(opt_val, (float,int)) and not np.isinf(opt_val) and
                        isinstance(base_val, (float,int)) and not np.isnan(base_val)):
                        if opt_val == 0.0:
                            # If the optimal is 0 => difference is effectively infinite
                            pct_diff = np.nan
                        else:
                            pct_diff = ((base_val - opt_val) / opt_val) * 100.0
                    else:
                        pct_diff = np.nan
                    percent_diff_dict[algo_name].append(pct_diff)

            # A.1) Plot line chart of percentage difference per step
            fig = plt.figure(figsize=(8,6))
            for algo_name in percent_diff_dict:
                steps = range(len(percent_diff_dict[algo_name]))
                plt.plot(steps, percent_diff_dict[algo_name], marker='o', markersize=3, label=algo_name)
            plt.title("Latency % Difference from Optimal (per Step)")
            plt.xlabel("Step")
            plt.ylabel("Percentage Over Optimal (%)")
            plt.legend()
            plt.grid(True)
            save_figure(fig, "percentage_diff_optimal_per_step")

            # B) Bar chart of average difference across steps
            avg_diff_dict = {}
            for algo_name in percent_diff_dict:
                valid_vals = [d for d in percent_diff_dict[algo_name]
                              if (isinstance(d, (float,int)) and not np.isnan(d))]
                if valid_vals:
                    avg_diff = float(np.mean(valid_vals))
                else:
                    avg_diff = np.nan
                avg_diff_dict[algo_name] = avg_diff

            # create bar chart
            fig = plt.figure(figsize=(8,6))
            algo_list = sorted(avg_diff_dict.keys())
            diff_vals = [avg_diff_dict[a] for a in algo_list]
            x_pos = np.arange(len(algo_list))
            plt.bar(x_pos, diff_vals, color="darkorange")
            plt.xticks(x_pos, algo_list, rotation=20)
            plt.title("Average Latency % Difference vs. Optimal (All Steps)")
            plt.ylabel("Avg ((Base - Opt)/Opt) * 100%")
            plt.grid(axis="y")
            save_figure(fig, "avg_percentage_diff_vs_optimal")

    print(f"Plots successfully saved in: {out_dir}")

if __name__ == "__main__":
    main()
