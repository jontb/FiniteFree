import os
import numpy as np
import matplotlib.pyplot as plt
from finitefree.orthogonal import unitary_hermite_polynomial
from PIL import Image

os.makedirs("visuals/assets", exist_ok=True)


def animate_unitary_trajectories():
    print("Generating Unitary Hermite Domain Coloring and trajectories animation...")
    d = 8
    t_vals = np.linspace(0.0, 5.0, 60)
    frame_images = []
    
    os.makedirs("visuals/assets/temp_frames", exist_ok=True)
    
    # 2D Grid for domain coloring
    res = 300
    x = np.linspace(-1.5, 1.5, res)
    y = np.linspace(-1.5, 1.5, res)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    for idx, t in enumerate(t_vals):
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        
        poly = unitary_hermite_polynomial(d, t)
        float_coeffs = [float(c) for c in poly.coeffs]
        
        # Evaluate polynomial on the complex grid
        W = np.polyval(float_coeffs, Z)
        phase = np.angle(W)
        
        # Plot domain coloring using twilight (cyclic phase colormap)
        im = ax.imshow(phase, extent=[-1.5, 1.5, -1.5, 1.5], cmap='twilight', origin='lower', alpha=0.9)
        
        # Plot unit circle in white
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), color='white', linestyle='--', lw=2.0, zorder=2)
        
        # Plot exact roots (eigenvalues)
        roots = poly.evaluate_roots_float64()
        ax.scatter(np.real(roots), np.imag(roots), color='black', edgecolor='white', s=90, zorder=3, linewidths=1.5, label=f"Roots (t={t:.2f})")
        
        # Add radian labels near the unit circle
        labels = [
            (0.0, r"$0$"), (np.pi/4, r"$\pi/4$"), (np.pi/2, r"$\pi/2$"), (3*np.pi/4, r"$3\pi/4$"),
            (np.pi, r"$\pi$"), (-3*np.pi/4, r"$-3\pi/4$"), (-np.pi/2, r"$-\pi/2$"), (-np.pi/4, r"$-\pi/4$")
        ]
        for ang, txt in labels:
            ax.text(1.23 * np.cos(ang), 1.23 * np.sin(ang), txt, color='white', fontsize=10, 
                    ha='center', va='center', fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.4, boxstyle='round,pad=0.2', edgecolor='none'))
            
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        
        ax.set_title(r"Unitary Hermite Polynomial Domain Coloring & Roots on $\mathbb{T}$", fontsize=11, fontweight='bold', pad=12)
        ax.set_xlabel("Re(z)")
        ax.set_ylabel("Im(z)")
        ax.legend(loc="lower right", framealpha=0.8)
        
        plt.tight_layout()
        
        frame_path = f"visuals/assets/temp_frames/frame_{idx:03d}.png"
        plt.savefig(frame_path, dpi=120)
        plt.close()
        
        frame_images.append(Image.open(frame_path))
        
    gif_path = "visuals/assets/unitary_trajectories.gif"
    if frame_images:
        frame_images[0].save(
            gif_path,
            save_all=True,
            append_images=frame_images[1:],
            duration=80,
            loop=0
        )
    print(f"Saved domain-colored animation to {gif_path}")
    
    for img in frame_images:
        img.close()
        
    # Cleanup temp frames
    for idx in range(len(t_vals)):
        try:
            os.remove(f"visuals/assets/temp_frames/frame_{idx:03d}.png")
        except OSError:
            pass
    try:
        os.rmdir("visuals/assets/temp_frames")
    except OSError:
        pass


if __name__ == "__main__":
    animate_unitary_trajectories()
