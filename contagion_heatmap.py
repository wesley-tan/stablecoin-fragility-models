"""
Contagion Heatmap Template
===========================

For Topic 2: Cross-issuer spillovers

Deliverable: Network contagion heatmap
"1σ USDC shock → x% DAI depeg"
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from contagion_network import create_usdc_dai_usdt_network


def create_contagion_heatmap(shock_sizes: list = [0.05, 0.10, 0.15, 0.20],
                            save_path: str = 'contagion_heatmap.png'):
    """
    Generate contagion matrix heatmap.
    
    C[i,j] = depeg in coin j when shock to coin i
    
    Args:
        shock_sizes: List of shock sizes to test
        save_path: Where to save figure
    """
    network = create_usdc_dai_usdt_network()
    coin_names = network.coin_names
    n_coins = len(coin_names)
    
    print("\n🔥 Computing contagion matrix...")
    print(f"  Testing shock sizes: {[f'{s*100:.0f}%' for s in shock_sizes]}")
    
    # Create subplots for each shock size
    n_shocks = len(shock_sizes)
    fig, axes = plt.subplots(1, n_shocks, figsize=(4*n_shocks, 4))
    
    if n_shocks == 1:
        axes = [axes]
    
    for idx, shock_size in enumerate(shock_sizes):
        print(f"\n  Shock size: {shock_size*100:.0f}%")
        
        # Compute contagion matrix
        contagion = np.zeros((n_coins, n_coins))
        
        for i, shock_coin in enumerate(coin_names):
            print(f"    Simulating {shock_coin} shock...")
            result = network.simulate_shock(shock_coin, shock_size, n_rounds=3)
            
            # Extract final depegs
            for j, target_coin in enumerate(coin_names):
                depeg_bps = result['summary'][target_coin]['max_depeg_bps']
                contagion[i, j] = depeg_bps
        
        # Plot heatmap
        ax = axes[idx]
        
        # Custom colormap: white → yellow → orange → red
        colors = ['white', 'lightyellow', 'yellow', 'orange', 'red', 'darkred']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('contagion', colors, N=n_bins)
        
        im = ax.imshow(contagion, cmap=cmap, aspect='auto', 
                      vmin=0, vmax=min(1000, np.max(contagion) * 1.1))
        
        # Labels
        ax.set_xticks(range(n_coins))
        ax.set_yticks(range(n_coins))
        ax.set_xticklabels(coin_names)
        ax.set_yticklabels(coin_names)
        
        ax.set_xlabel('Depeg in →', fontsize=10, fontweight='bold')
        ax.set_ylabel('Shock to ↓', fontsize=10, fontweight='bold')
        ax.set_title(f'Shock: {shock_size*100:.0f}%', fontsize=12, fontweight='bold')
        
        # Annotate cells
        for i in range(n_coins):
            for j in range(n_coins):
                val = contagion[i, j]
                if val >= 1000:
                    text = f'{val/1000:.1f}k'
                else:
                    text = f'{val:.0f}'
                
                # Color: white text on dark, black on light
                color = 'white' if val > 500 else 'black'
                ax.text(j, i, text, ha='center', va='center',
                       color=color, fontsize=9, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Depeg (bps)', fontsize=9)
    
    plt.suptitle('Cross-Issuer Contagion Matrix', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")
    
    return contagion


def impulse_response_plot(shock_coin: str = 'USDC',
                         shock_size: float = 0.115,
                         save_path: str = 'impulse_response.png'):
    """
    IRF plot: "1σ USDC shock → x% DAI depeg over time"
    
    VAR-style impulse response.
    """
    network = create_usdc_dai_usdt_network()
    
    print(f"\n📉 Computing IRF: {shock_coin} → all coins")
    
    irf = network.impulse_response(shock_coin, shock_size, horizon=8)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Panel 1: Depegs
    for coin in network.coin_names:
        depegs = irf['responses'][coin]['depeg_bps']
        periods = range(len(depegs))
        
        style = '-' if coin == shock_coin else '--'
        lw = 2.5 if coin == shock_coin else 2
        
        ax1.plot(periods, depegs, style, lw=lw, label=coin, marker='o')
    
    ax1.set_ylabel('Depeg (bps)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Impulse Response: {shock_coin} Shock ({shock_size*100:.1f}%)',
                 fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Redemptions
    for coin in network.coin_names:
        redemptions = irf['responses'][coin]['redemption_pct']
        periods = range(len(redemptions))
        
        style = '-' if coin == shock_coin else '--'
        lw = 2.5 if coin == shock_coin else 2
        
        ax2.plot(periods, redemptions, style, lw=lw, label=coin, marker='s')
    
    ax2.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Redemptions (%)', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    
    return irf


def network_visualization(save_path: str = 'network_graph.png'):
    """
    Simple network graph showing DEX pool connections.
    """
    network = create_usdc_dai_usdt_network()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Node positions (triangle)
    positions = {
        'USDC': np.array([0.5, 0.9]),
        'DAI': np.array([0.2, 0.2]),
        'USDT': np.array([0.8, 0.2]),
    }
    
    # Draw edges (DEX pools)
    for pool in network.dex_pools:
        pos_a = positions[pool.coin_a]
        pos_b = positions[pool.coin_b]
        
        # Edge thickness by pool size
        pool_size = pool.reserve_a + pool.reserve_b
        thickness = np.log(pool_size + 1) / 2
        
        ax.plot([pos_a[0], pos_b[0]], [pos_a[1], pos_b[1]], 
               'gray', lw=thickness, alpha=0.6, zorder=1)
        
        # Edge label (pool size)
        mid = (pos_a + pos_b) / 2
        ax.text(mid[0], mid[1], f'${pool_size:.0f}M',
               fontsize=8, ha='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Draw nodes
    for coin_name, pos in positions.items():
        coin = network.stablecoins[coin_name]
        
        # Node size by supply
        size = np.sqrt(coin.supply) * 50
        
        # Color by lambda (liquid share)
        color_val = coin.lambda_tier1
        color = plt.cm.RdYlGn(color_val)
        
        ax.scatter(pos[0], pos[1], s=size, c=[color], 
                  edgecolors='black', linewidths=2, zorder=2)
        
        # Label
        ax.text(pos[0], pos[1] + 0.08, coin_name,
               fontsize=14, fontweight='bold', ha='center')
        ax.text(pos[0], pos[1] - 0.08, f'${coin.supply:.0f}B\nλ={coin.lambda_tier1*100:.0f}%',
               fontsize=9, ha='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Stablecoin Contagion Network\n(Node size = supply, edge = DEX pool)',
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")


if __name__ == "__main__":
    print("=" * 70)
    print("CONTAGION ANALYSIS: NETWORK HEATMAPS & IRFs")
    print("=" * 70)
    
    # 1. Contagion heatmap
    print("\n1. Generating contagion heatmap...")
    create_contagion_heatmap(shock_sizes=[0.10, 0.15, 0.20])
    
    # 2. Impulse response
    print("\n2. Generating IRF plot...")
    impulse_response_plot('USDC', 0.115)
    
    # 3. Network graph
    print("\n3. Generating network visualization...")
    network_visualization()
    
    print("\n" + "=" * 70)
    print("✅ All contagion deliverables generated!")
    print("=" * 70)
    print("\nFiles created:")
    print("  - contagion_heatmap.png (matrix)")
    print("  - impulse_response.png (VAR-style IRF)")
    print("  - network_graph.png (topology)")


