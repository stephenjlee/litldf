import os, sys, copy, re
from dotenv import load_dotenv, find_dotenv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams.update({'font.size': 20})
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.stats import nbinom
import matplotlib.cm as cm

load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))


def create_gif_with_name_match(subfolder,
                               input_folder_path,
                               output_gif_path,
                               font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'):

    if subfolder == 'rotate':
        print('asdf')
        png_files = [f for f in os.listdir(input_folder_path) if f.endswith('.png')]
        def sort_key(file):
            return int(file.split('_')[0])
        # sort files
        png_files = sorted(png_files, key=sort_key)

        max_iter = int(png_files[-1].split('_')[0])

    else:
        # Get all png files in the input folder
        png_files = [f for f in os.listdir(input_folder_path) if f.endswith('.png')]
        heatmap_files = [file for file in png_files if re.fullmatch(f'\d+\_{subfolder}.png', file)]
        # sort the files
        heatmap_files_sorted = sorted(heatmap_files, key=lambda file: int(re.search(r'\d+', file).group()))
        png_files = heatmap_files_sorted

        max_iter = int(re.search(r'\d+', heatmap_files_sorted[-1]).group())

    # Open the first image and draw text on it
    first_image_path = os.path.join(input_folder_path, png_files[0])
    first_image = Image.open(first_image_path)
    draw = ImageDraw.Draw(first_image)
    font = ImageFont.truetype(font_path, size=20)
    iter = re.search('\d+', png_files[0]).group()
    text = f"MH iteration {iter} of {max_iter}"
    text_width, _ = draw.textsize(text, font=font)
    position = (first_image.width - text_width - 10, 10)
    draw.text(position, text, fill="black", font=font)

    # Open the rest of the images, draw text on them, and add them to a list
    other_images = []
    for file in png_files[1:]:
        image_path = os.path.join(input_folder_path, file)
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_path, size=20)
        iter = re.search('\d+', file).group()
        text = f"MH iteration {iter} of {max_iter}"
        text_width, _ = draw.textsize(text, font=font)
        position = (first_image.width - text_width - 10, 10)
        draw.text(position, text, fill="black", font=font)
        other_images.append(image)

    # Save the first image to a gif and add the other images as additional frames
    first_image.save(output_gif_path, 'GIF', append_images=other_images, save_all=True, duration=100, loop=0)


def create_gif(input_folder_path, output_gif_path):
    # Get all png files in the input folder
    png_files = [f for f in os.listdir(input_folder_path) if f.endswith('.png')]

    png_files.sort()

    png_files = copy.deepcopy(png_files)

    # Use the first image to start the gif
    first_image_path = os.path.join(input_folder_path, png_files[0])
    first_image = Image.open(first_image_path)

    # Open the rest of the images and add them to a list
    other_images = []
    for file in png_files[1:]:
        image_path = os.path.join(input_folder_path, file)
        image = Image.open(image_path)
        other_images.append(image)

    # Save the first image to a gif and add the other images as additional frames
    first_image.save(output_gif_path, 'GIF', append_images=other_images, save_all=True, duration=100, loop=0)


def convert_a_b_to_n_p(a, b):
    r_wiki = a
    p_wiki = 1. / (b + 1.)
    n_wiki_alt = r_wiki
    p_wiki_alt = 1 - p_wiki
    n = n_wiki_alt
    p = p_wiki_alt
    return n, p


def neg_binom_hist(fig, ax, a1, a2, a3, b1, b2, b3, num_points, bins, colormap, min_alpha, max_alpha):
    np.random.seed(1)

    n1, p1 = convert_a_b_to_n_p(a1, b1)
    n2, p2 = convert_a_b_to_n_p(a2, b2)
    n3, p3 = convert_a_b_to_n_p(a3, b3)

    x = nbinom.rvs(n1, p1, size=num_points)
    y = nbinom.rvs(n2, p2, size=num_points)
    z = nbinom.rvs(n3, p3, size=num_points)

    hist, edges = np.histogramdd([x, y, z], bins=bins)

    x_edges_mid = edges[0][:-1] + np.diff(edges[0]) / 2
    y_edges_mid = edges[1][:-1] + np.diff(edges[1]) / 2
    z_edges_mid = edges[2][:-1] + np.diff(edges[2]) / 2

    x_mid, y_mid, z_mid = np.meshgrid(x_edges_mid, y_edges_mid, z_edges_mid, indexing='ij')

    x_mid = x_mid.flatten()
    y_mid = y_mid.flatten()
    z_mid = z_mid.flatten()
    hist = hist.flatten() / num_points * 100.

    norm = plt.Normalize(hist.min(), hist.max())
    colors = colormap(norm(hist))

    alphas = norm(hist) * (max_alpha - min_alpha) + min_alpha
    colors[..., -1] = alphas

    sc = ax.scatter(x_mid, y_mid, z_mid, c=colors, s=30)

    axins = inset_axes(ax,
                       width="5%",  # width = 5% of parent_bbox width
                       height="50%",  # height : 50%
                       loc='lower left',
                       bbox_to_anchor=(1.45, 0.4, 1, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0,
                       )

    colorbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), cax=axins)

    # colorbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax, shrink=0.5, pad=0.05)
    colorbar.set_label('predicted likelihood (%)')  # Add label to the colorbar
    colorbar.ax.yaxis.set_label_coords(-2.0, 0.5)


def plot_simplex_for_a013(azim=-30,
                          n_A=30,
                          n_B=30,
                          n_A_2=40,
                          n_B_2=40,
                          a1=30,
                          a2=20,
                          a3=70,
                          b1=2,
                          b2=0.5,
                          b3=5,
                          y_samps_0=None,
                          y_samps_1=None,
                          output_folder_path='plot_constraints',
                          fig_prefix='',
                          subfolder='subfolder'
                          ):
    ###############################################
    # config
    ###############################################
    ###############################################
    # setup
    ###############################################
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    # Draw Negative Binomial distribution

    num_points = 10000000
    bins = 30
    colormap = cm.viridis
    min_alpha = 0.0
    max_alpha = 1.0
    neg_binom_hist(fig, ax, a1, a2, a3, b1, b2, b3, num_points, bins, colormap, min_alpha, max_alpha)

    # Make the axes of the plot equal
    ax.set_box_aspect([1, 1, 1])

    # Set the limits of x, y, and z to be from 0 to n_A + n_B
    ax.set_xlim(0, n_A_2 + n_B_2)
    ax.set_ylim(0, n_A_2 + n_B_2)
    ax.set_zlim(0, n_A_2 + n_B_2)

    ############################################
    # plot simplex 1

    vertices = np.array([
        [n_A + n_B, 0, 0],
        [0, n_A + n_B, 0],
        [0, 0, n_A + n_B]])

    # Plot vertices of simplex
    ax.scatter(vertices[0, 0], vertices[0, 1], vertices[0, 2], color='b')
    ax.scatter(vertices[1, 0], vertices[1, 1], vertices[1, 2], color='b')
    ax.scatter(vertices[2, 0], vertices[2, 1], vertices[2, 2], color='b')

    # Plot edges of simplex
    for i in range(3):
        for j in range(i + 1, 3):
            ax.plot(*zip(vertices[i], vertices[j]), color='darkgray', linewidth=0.1)

    # Plot the face of the simplex
    faces = [vertices]
    ax.add_collection3d(Poly3DCollection(faces, facecolors='gray', linewidths=1, edgecolors='gray', alpha=0.05))

    ############################################
    # plot simplex 2

    vertices = np.array([
        [n_A_2 + n_B_2, 0, 0],
        [0, n_A_2 + n_B_2, 0],
        [0, 0, n_A_2 + n_B_2]])

    # Plot vertices of simplex
    ax.scatter(vertices[0, 0], vertices[0, 1], vertices[0, 2], color='b')
    ax.scatter(vertices[1, 0], vertices[1, 1], vertices[1, 2], color='b')
    ax.scatter(vertices[2, 0], vertices[2, 1], vertices[2, 2], color='b')

    # Plot edges of simplex
    for i in range(3):
        for j in range(i + 1, 3):
            ax.plot(*zip(vertices[i], vertices[j]), color='darkgray', linewidth=0.1)

    # Plot the face of the simplex
    faces = [vertices]
    ax.add_collection3d(Poly3DCollection(faces, facecolors='gray', linewidths=1, edgecolors='gray', alpha=0.05))

    ############################################
    # plot scatterplots
    ############################################

    # plot scatter
    scatter1 = ax.scatter(y_samps_0[:, 0], y_samps_0[:, 1], y_samps_0[:, 2], color='b', alpha=0.2,
                          label='subgraph 1 MH samples')

    # plot scatter
    scatter2 = ax.scatter(y_samps_1[:, 0], y_samps_1[:, 1], y_samps_1[:, 2], color='g', alpha=0.2,
                          label='subgraph 2 MH samples')

    ############################################
    # plot feasible region 1

    # plot the face of the feasible region
    vertices_feas = np.array([
        [0, 0, n_A, n_A],
        [n_A + n_B, n_A, 0, n_B],
        [0, n_B, n_B, 0]]).T
    # Plot the face of the simplex
    faces_feas = [vertices_feas]
    ax.add_collection3d(
        Poly3DCollection(faces_feas, facecolors='blue', linewidths=0.5, edgecolors='blue', alpha=.25))

    ############################################
    # plot feasible region 1

    # plot the face of the feasible region
    vertices_feas = np.array([
        [0, 0, n_A_2, n_A_2],
        [n_A_2 + n_B_2, n_A_2, 0, n_B_2],
        [0, n_B_2, n_B_2, 0]]).T
    # Plot the face of the simplex
    faces_feas = [vertices_feas]
    ax.add_collection3d(
        Poly3DCollection(faces_feas, facecolors='green', linewidths=0.5, edgecolors='green', alpha=.25))

    ###############################################
    # plot axes and legend
    ###############################################
    # Rotate the plot so the origin is facing away from the viewer
    # ax.view_init(elev=10, azim=30)
    ax.view_init(elev=10, azim=azim)

    # label axes
    ax.set_xlabel(r"$y_1^\top 1$", labelpad=12)
    ax.set_ylabel(r"$y_2^\top 1$", labelpad=12)
    ax.set_zlabel(r"$y_3^\top 1$", labelpad=12)
    # Create legend
    import matplotlib.patches as mpatches

    # [blue_patch, purple_patch, green_line, green_feas_region]
    handles = []
    blue_feas_region = mpatches.Patch(color='blue', label='feasible region, subgraph 1', alpha=0.3)
    handles.append(blue_feas_region)
    green_feas_region = mpatches.Patch(color='green', label='feasible region, subgraph 2', alpha=0.3)
    handles.append(green_feas_region)

    # append scatterplots to handles
    handles.append(scatter1)
    handles.append(scatter2)

    legend = ax.legend(handles=handles, loc='center left',
                       bbox_to_anchor=(1.05, 0.15))
    # Increase the size of the scatterplot legend icons
    for handle in legend.legendHandles[-2:]:  # assuming the last two handles are for scatterplots
        handle._sizes = [100]  # change size as per your needs

    # Increase the alpha of the scatterplot legend icons to make them bolder
    for handle in legend.legendHandles[-2:]:  # assuming the last two handles are for scatterplots
        handle.set_alpha(1)  # set alpha as per your needs

    output_dir = os.path.join(output_folder_path, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    # ax.tight_layout()
    plt.subplots_adjust(left=-0.33, right=0.9)

    ax.set_title(f'Multiple feasible regions and \n neural network training')

    plt.savefig(os.path.join(output_dir, f'{fig_prefix}_{str(int(azim)).zfill(3)}.png'))
    plt.close()
