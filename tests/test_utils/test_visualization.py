import pytest
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import json
from typing import Union

from src.utils.visualization import VisualizationManager

@pytest.fixture
def visualization_manager():
    """Create visualization manager"""
    with tempfile.TemporaryDirectory() as temp_dir:
        return VisualizationManager(
            output_dir=temp_dir,
            style='seaborn-whitegrid'
        )

@pytest.fixture
def sample_metrics_data():
    """Create sample metrics data"""
    data = []
    devices = ['device_0', 'device_1']
    for step in range(10):
        for device_id in devices:
            data.append({
                'step': step,
                'device_id': device_id,
                'memory_utilization': np.random.uniform(0, 1),
                'compute_utilization': np.random.uniform(0, 1)
            })
    return pd.DataFrame(data)

@pytest.fixture
def sample_network():
    """Create sample network graph"""
    G = nx.Graph()
    G.add_nodes_from(['device_0', 'device_1', 'device_2'])
    G.add_edges_from([('device_0', 'device_1'), ('device_1', 'device_2')])
    return G

class TestVisualizationManager:
    def test_initialization(self, visualization_manager):
        """Test visualization manager initialization"""
        assert visualization_manager.output_dir.exists()
        plt.style.use('seaborn-whitegrid')  # Should not raise error

    def test_save_plot(self, visualization_manager):
        """Test plot saving functionality"""
        # Create a simple plot
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        # Save in multiple formats
        visualization_manager.save_plot(
            fig=fig,
            name="test_plot",
            formats=['png', 'pdf']
        )
        
        # Check files created
        assert (Path(visualization_manager.output_dir) / "test_plot.png").exists()
        assert (Path(visualization_manager.output_dir) / "test_plot.pdf").exists()
        
        plt.close(fig)

    def test_resource_utilization_plot(self, visualization_manager, sample_metrics_data):
        """Test resource utilization plotting"""
        device_ids = sample_metrics_data['device_id'].unique()
        
        fig = visualization_manager.plot_resource_utilization(
            metrics_data=sample_metrics_data,
            device_ids=device_ids
        )
        
        # Check plot components
        assert len(fig.axes) == 2  # Should have two subplots
        
        # Verify axes labels
        assert fig.axes[0].get_ylabel() == 'Memory Utilization (%)'
        assert fig.axes[1].get_ylabel() == 'Compute Utilization (%)'
        
        plt.close(fig)

    def test_latency_comparison_plot(self, visualization_manager):
        """Test latency comparison plotting"""
        algorithms = ['algo1', 'algo2']
        latencies = {
            'algo1': [100, 150, 200],
            'algo2': [90, 140, 180]
        }
        workload_types = ['small', 'medium', 'large']
        
        fig = visualization_manager.plot_latency_comparison(
            algorithms=algorithms,
            latencies=latencies,
            workload_types=workload_types
        )
        
        # Check plot components
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        
        assert ax.get_xlabel() == 'Workload Type'
        assert ax.get_ylabel() == 'Average Latency (ms)'
        
        plt.close(fig)

    def test_network_topology_plot(self, visualization_manager, sample_network):
        """Test network topology plotting"""
        node_colors = {node: 'blue' for node in sample_network.nodes()}
        edge_weights = {edge: 1.0 for edge in sample_network.edges()}
        
        fig = visualization_manager.plot_network_topology(
            graph=sample_network,
            node_colors=node_colors,
            edge_weights=edge_weights
        )
        
        # Check plot components
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert not ax.get_xticks().size  # Should be no axis ticks
        
        plt.close(fig)

    def test_workload_statistics_plot(self, visualization_manager):
        """Test workload statistics plotting"""
        workload_data = pd.DataFrame({
            'model_type': ['small', 'medium', 'large'] * 10,
            'sequence_length': np.random.randint(128, 512, 30),
            'generation_steps': np.random.randint(32, 128, 30),
            'memory_requirement': np.random.uniform(1, 10, 30),
            'compute_requirement': np.random.uniform(10, 100, 30)
        })
        
        fig = visualization_manager.plot_workload_statistics(workload_data)
        
        # Check plot components
        assert len(fig.axes) == 4  # Should have four subplots
        
        # Verify subplot types
        assert isinstance(fig.axes[0], plt.Axes)  # Sequence length distribution
        assert isinstance(fig.axes[1], plt.Axes)  # Generation steps distribution
        assert isinstance(fig.axes[2], plt.Axes)  # Memory requirements
        assert isinstance(fig.axes[3], plt.Axes)  # Compute requirements
        
        plt.close(fig)

def test_performance_report_creation(visualization_manager):
    """Test performance report creation"""
    metrics_data = pd.DataFrame({
        'step': range(10),
        'latency': np.random.uniform(10, 100, 10),
        'memory_utilization': np.random.uniform(0, 1, 10),
        'compute_utilization': np.random.uniform(0, 1, 10),
        'data_transferred': np.random.uniform(0, 10, 10)
    })
    
    output_file = Path(visualization_manager.output_dir) / "performance_report.json"
    visualization_manager.create_performance_report(metrics_data, output_file)
    
    # Check report file
    assert output_file.exists()
    
    # Verify report contents
    with open(output_file) as f:
        report = json.load(f)
        assert 'timestamp' in report
        assert 'summary_statistics' in report
        assert 'resource_utilization' in report
        assert 'communication_statistics' in report

def test_plot_all_metrics(visualization_manager, tmp_path):
    """Test plotting all metrics"""
    # Create sample metrics data
    metrics_data = pd.DataFrame({
        'step': range(10),
        'device_id': ['device_0'] * 10,
        'latency': np.random.uniform(10, 100, 10),
        'memory_utilization': np.random.uniform(0, 1, 10),
        'compute_utilization': np.random.uniform(0, 1, 10),
        'data_transferred': np.random.uniform(0, 10, 10),
        'event_type': ['computation'] * 5 + ['communication'] * 5
    })
    
    # Save metrics data
    metrics_file = tmp_path / 'metrics.jsonl'
    metrics_data.to_json(metrics_file, orient='records', lines=True)
    
    # Plot all metrics
    plot_all_metrics(
        metrics_dir=tmp_path,
        output_dir=visualization_manager.output_dir
    )
    
    # Check output files
    output_dir = Path(visualization_manager.output_dir)
    assert (output_dir / 'resource_utilization.png').exists()
    assert (output_dir / 'communication_patterns.png').exists()
    assert (output_dir / 'performance_report.json').exists()

def test_visualization_error_handling(visualization_manager):
    """Test error handling in visualization functions"""
    # Test with empty data
    empty_df = pd.DataFrame()
    
    with pytest.raises(ValueError):
        visualization_manager.plot_resource_utilization(empty_df, [])
        
    with pytest.raises(ValueError):
        visualization_manager.plot_workload_statistics(empty_df)
        
    # Test with invalid network graph
    invalid_graph = nx.Graph()  # Empty graph
    
    with pytest.raises(ValueError):
        visualization_manager.plot_network_topology(
            invalid_graph,
            node_colors={},
            edge_weights={}
        )

def plot_all_metrics(metrics_dir: Union[str, Path], output_dir: Union[str, Path]) -> None:
    """Helper function to plot all metrics"""
    metrics_file = Path(metrics_dir) / 'metrics.jsonl'
    viz = VisualizationManager(output_dir)
    
    # Load metrics data
    metrics_data = pd.read_json(metrics_file, lines=True)
    
    # Plot resource utilization
    device_ids = metrics_data['device_id'].unique()
    fig = viz.plot_resource_utilization(metrics_data, device_ids)
    viz.save_plot(fig, 'resource_utilization')
    plt.close(fig)
    
    # Plot communication patterns
    comm_data = metrics_data[metrics_data['event_type'] == 'communication']
    if not comm_data.empty:
        fig = viz.plot_communication_patterns(comm_data)
        viz.save_plot(fig, 'communication_patterns')
        plt.close(fig)
    
    # Create performance report
    viz.create_performance_report(
        metrics_data,
        Path(output_dir) / 'performance_report.json'
    )

def test_custom_visualization_options(visualization_manager, sample_metrics_data):
    """Test customization of visualization options"""
    device_ids = sample_metrics_data['device_id'].unique()
    
    # Test custom figure size
    fig = visualization_manager.plot_resource_utilization(
        metrics_data=sample_metrics_data,
        device_ids=device_ids,
        figsize=(15, 10)
    )
    assert fig.get_size_inches().tolist() == [15, 10]
    plt.close(fig)
    
    # Test custom color scheme
    fig = visualization_manager.plot_latency_comparison(
        algorithms=['algo1', 'algo2'],
        latencies={'algo1': [1, 2], 'algo2': [2, 3]},
        workload_types=['small', 'large'],
        colors=['red', 'blue']
    )
    plt.close(fig)

def test_plot_saving_options(visualization_manager):
    """Test different plot saving options"""
    # Create a simple plot
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    
    # Test different DPI settings
    visualization_manager.save_plot(
        fig=fig,
        name="high_dpi",
        formats=['png'],
        dpi=600
    )
    
    # Test with tight layout
    visualization_manager.save_plot(
        fig=fig,
        name="tight_layout",
        formats=['png'],
        tight_layout=True
    )
    
    # Check file sizes (high DPI should be larger)
    high_dpi_size = (Path(visualization_manager.output_dir) / "high_dpi.png").stat().st_size
    normal_size = (Path(visualization_manager.output_dir) / "tight_layout.png").stat().st_size
    assert high_dpi_size > normal_size
    
    plt.close(fig)

def test_animation_creation(visualization_manager, sample_metrics_data):
    """Test creation of animated visualizations"""
    device_ids = sample_metrics_data['device_id'].unique()
    
    animation = visualization_manager.create_resource_animation(
        metrics_data=sample_metrics_data,
        device_ids=device_ids,
        interval=200  # ms between frames
    )
    
    # Save animation
    animation_path = Path(visualization_manager.output_dir) / "resource_animation.gif"
    animation.save(str(animation_path))
    
    assert animation_path.exists()

def test_interactive_plots(visualization_manager, sample_network):
    """Test creation of interactive plots"""
    node_colors = {node: 'blue' for node in sample_network.nodes()}
    edge_weights = {edge: 1.0 for edge in sample_network.edges()}
    
    # Create interactive network plot
    html_path = Path(visualization_manager.output_dir) / "interactive_network.html"
    visualization_manager.create_interactive_network_plot(
        graph=sample_network,
        node_colors=node_colors,
        edge_weights=edge_weights,
        output_path=html_path
    )
    
    assert html_path.exists()
    
def test_report_formats(visualization_manager, sample_metrics_data):
    """Test different performance report formats"""
    # Test JSON format
    json_path = Path(visualization_manager.output_dir) / "report.json"
    visualization_manager.create_performance_report(
        metrics_data=sample_metrics_data,
        output_path=json_path,
        format='json'
    )
    assert json_path.exists()
    
    # Test HTML format
    html_path = Path(visualization_manager.output_dir) / "report.html"
    visualization_manager.create_performance_report(
        metrics_data=sample_metrics_data,
        output_path=html_path,
        format='html'
    )
    assert html_path.exists()
    
    # Test PDF format (requires LaTeX)
    try:
        pdf_path = Path(visualization_manager.output_dir) / "report.pdf"
        visualization_manager.create_performance_report(
            metrics_data=sample_metrics_data,
            output_path=pdf_path,
            format='pdf'
        )
        assert pdf_path.exists()
    except ImportError:
        pytest.skip("LaTeX not available for PDF generation")

def test_plot_style_consistency(visualization_manager):
    """Test consistency of plot styling across different plot types"""
    # Create sample data
    metrics_data = pd.DataFrame({
        'step': range(10),
        'device_id': ['device_0'] * 10,
        'memory_utilization': np.random.uniform(0, 1, 10),
        'compute_utilization': np.random.uniform(0, 1, 10)
    })
    
    # Generate different types of plots
    plots = []
    
    # Resource utilization plot
    fig1 = visualization_manager.plot_resource_utilization(
        metrics_data=metrics_data,
        device_ids=['device_0']
    )
    plots.append(fig1)
    
    # Latency comparison plot
    fig2 = visualization_manager.plot_latency_comparison(
        algorithms=['algo1'],
        latencies={'algo1': [1, 2, 3]},
        workload_types=['small', 'medium', 'large']
    )
    plots.append(fig2)
    
    # Check style consistency
    for fig in plots:
        for ax in fig.axes:
            assert ax.get_facecolor() == (1.0, 1.0, 1.0, 1.0)  # white background
            assert ax.grid(True)  # grid should be enabled
        plt.close(fig)
        
def test_large_dataset_handling(visualization_manager):
    """Test visualization performance with large datasets"""
    # Create large dataset
    num_points = 10000
    large_metrics = pd.DataFrame({
        'step': range(num_points),
        'device_id': ['device_0'] * num_points,
        'memory_utilization': np.random.uniform(0, 1, num_points),
        'compute_utilization': np.random.uniform(0, 1, num_points)
    })
    
    # Test plotting with downsampling
    fig = visualization_manager.plot_resource_utilization(
        metrics_data=large_metrics,
        device_ids=['device_0'],
        downsample=True,
        max_points=1000
    )
    
    # Verify number of points plotted is reduced
    line = fig.axes[0].get_lines()[0]
    assert len(line.get_xdata()) <= 1000
    
    plt.close(fig)