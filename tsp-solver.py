import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import networkx as nx
import time
import itertools
from typing import List, Tuple

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"assets")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

# TSP Solver: Toda la lógica para resolver el problema del agente viajero
class TSPSolver:
    def __init__(self):
        self.distances = None
        self.n = 0
        self.coordinates = None

    def generate_random_matrix(self, n: int) -> None:
        self.n = n
        self.distances = np.random.randint(1, 100, size=(n, n))
        np.fill_diagonal(self.distances, 0)
        self.distances = (self.distances + self.distances.T) // 2  # Hacer simétrica la matriz

    # Algoritmo de fuerza bruta
    def solve_tsp_brute_force(self) -> Tuple[List[int], int]:
        cities = range(self.n)
        shortest_path = None
        min_distance = float('inf')
        
        for path in itertools.permutations(cities):
            distance = sum(self.distances[path[i]][path[i+1]] for i in range(self.n-1))
            distance += self.distances[path[-1]][path[0]]  # Regresar al punto de inicio
            if distance < min_distance:
                min_distance = distance
                shortest_path = path
        
        return list(shortest_path), min_distance
    
    # Algoritmo de nearest neighbor con mejora 2-opt
    
    # Nearest Neighbor: Selecciona el vecino más cercano en cada paso
    def nearest_neighbor(self) -> List[int]:
        unvisited = set(range(1, self.n))
        path = [0]
        while unvisited:
            last = path[-1]
            next_city = min(unvisited, key=lambda city: self.distances[last][city])
            path.append(next_city)
            unvisited.remove(next_city)
        return path

    # 2-opt: Intercambia dos aristas para mejorar la solución
    def two_opt(self, tour):
        improved = True
        best_distance = self.calculate_total_distance(tour)
        while improved:
            improved = False
            for i in range(1, len(tour) - 1):
                for j in range(i + 1, len(tour)):
                    new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
                    new_distance = self.calculate_total_distance(new_tour)
                    if new_distance < best_distance:
                        tour = new_tour
                        best_distance = new_distance
                        improved = True
                        break
                if improved:
                    break
        return tour

    # Algoritmo de nearest neighbor con mejora 2-opt
    def solve_tsp_heuristic(self) -> Tuple[List[int], int]:
        initial_tour = self.nearest_neighbor()
        optimized_tour = self.two_opt(initial_tour)
        total_distance = self.calculate_total_distance(optimized_tour)
        return optimized_tour, total_distance

    # Definición de los métodos de resolución

    # Depende del valor de N
    def solve_tsp(self) -> Tuple[List[int], int, float, str]:
        start_time = time.time()
        if self.n < 10:
            path, distance = self.solve_tsp_brute_force()
            algorithm = "Fuerza Bruta"
        else:
            path, distance = self.solve_tsp_heuristic()
            algorithm = "Nearest Neighbor con 2-opt"
        end_time = time.time()
        execution_time = end_time - start_time
        return path, distance, execution_time, algorithm

    def calculate_total_distance(self, tour):
        return sum(self.distances[tour[i]][tour[i+1]] for i in range(len(tour)-1)) + self.distances[tour[-1]][tour[0]]

    # Visualización de los resultados en un grafo interactivo
    def visualize_interactive_graph(self):
        G = nx.Graph()
        for i in range(self.n):
            for j in range(i+1, self.n):
                G.add_edge(i, j, weight=self.distances[i][j])

        pos = nx.spring_layout(G)
        
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                size=40,
                color='white',
                line_width=1),
            textfont=dict(
                size=12,
                color='white'
            ))

        node_adjacencies = []
        node_text = []
        for node in G.nodes():
            node_adjacencies.append(len(list(G.neighbors(node))))
            node_text.append(f'C. {node}')

        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text

        annotations = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            annotations.append(dict(
                x=(x0 + x1) / 2,
                y=(y0 + y1) / 2,
                xref='x',
                yref='y',
                text=str(G.edges[edge]['weight']),
                showarrow=False,
                font=dict(size=10),
                bgcolor='rgba(255, 255, 255, 0.5)'
            ))

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Grafo Principal',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            annotations=annotations,
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )

        fig.show()

    # Visualización de la ruta óptima en un grafo interactivo
    def visualize_interactive_route(self, path: List[int], title: str = "Ruta Óptima"):
        G = nx.Graph()
        for i in range(self.n):
            for j in range(i+1, self.n):
                G.add_edge(i, j, weight=self.distances[i][j])

        pos = nx.spring_layout(G)
        
        edge_x = []
        edge_y = []
        for i in range(len(path)):
            start = path[i]
            end = path[(i + 1) % len(path)]
            x0, y0 = pos[start]
            x1, y1 = pos[end]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                size=40,
                color='white',
                line_width=1))

        node_text = [f'C. {i}' for i in range(self.n)]
        node_trace.text = node_text

        annotations = []
        total_distance = 0
        for i in range(len(path)):
            start = path[i]
            end = path[(i + 1) % len(path)]
            x0, y0 = pos[start]
            x1, y1 = pos[end]
            distance = self.distances[start][end]
            total_distance += distance
            annotations.append(dict(
                x=(x0 + x1) / 2,
                y=(y0 + y1) / 2,
                xref='x',
                yref='y',
                text=str(distance),
                showarrow=False,
                font=dict(size=10),
                bgcolor='rgba(255, 255, 255, 0.5)'
            ))

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=f'{title} (Distancia total: {total_distance})',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            annotations=annotations,
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )

        fig.show()

# GUI de la aplicación
class TSPGUI:
    def __init__(self, window):
        self.window = window
        self.window.geometry("800x650")
        self.window.configure(bg="#FFFFFF")
        self.window.title("El Agente Viajero")
        self.solver = TSPSolver()

        self.canvas = tk.Canvas(
            window,
            bg="#FFFFFF",
            height=650,
            width=800,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )
        self.canvas.place(x=0, y=0)

        self.image_image_1 = tk.PhotoImage(file=relative_to_assets("image_1.png"))
        self.canvas.create_image(400.0, 325.0, image=self.image_image_1)

        self.image_image_2 = tk.PhotoImage(file=relative_to_assets("image_2.png"))
        self.canvas.create_image(587.0, 324.0, image=self.image_image_2)

        self.image_image_3 = tk.PhotoImage(file=relative_to_assets("image_3.png"))
        self.canvas.create_image(223.0, 425.0, image=self.image_image_3)

        self.image_image_4 = tk.PhotoImage(file=relative_to_assets("image_4.png"))
        self.canvas.create_image(222.0, 119.0, image=self.image_image_4)

        self.entry_image_1 = tk.PhotoImage(file=relative_to_assets("entry_1.png"))
        entry_bg_1 = self.canvas.create_image(223.0, 345.0, image=self.entry_image_1)
        self.entry_1 = tk.Entry(
            bd=0,
            bg="#D9D9D9",
            fg="#000716",
            highlightthickness=0
        )
        self.entry_1.place(x=84.0, y=327.0, width=278.0, height=34.0)

        self.button_image_1 = tk.PhotoImage(file=relative_to_assets("button_1.png"))
        self.button_1 = tk.Button(
            image=self.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=self.generate_random_matrix,
            relief="flat"
        )
        self.button_1.place(x=77.0, y=388.0, width=292.0, height=47.0)

        self.button_image_2 = tk.PhotoImage(file=relative_to_assets("button_2.png"))
        self.button_2 = tk.Button(
            image=self.button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=self.open_manual_input_window,
            relief="flat"
        )
        self.button_2.place(x=77.0, y=447.0, width=292.0, height=47.0)

        self.button_image_3 = tk.PhotoImage(file=relative_to_assets("button_3.png"))
        self.button_3 = tk.Button(
            image=self.button_image_3,
            borderwidth=0,
            highlightthickness=0,
            command=self.solve_tsp,
            relief="flat"
        )
        self.button_3.place(x=77.0, y=506.0, width=292.0, height=47.0)

        self.canvas.create_text(
            77.0,
            301.0,
            anchor="nw",
            text="Tamaño de la matriz",
            fill="#000000",
            font=("Inter Medium", 16 * -1)
        )

    # Funciones de la GUI

    # Generar matriz aleatoria
    def generate_random_matrix(self):
        try:
            size = int(self.entry_1.get())
            if size < 5 or size > 16:
                raise ValueError("El tamaño debe estar entre 5 y 16")
            
            self.solver.generate_random_matrix(size)
            messagebox.showinfo("Éxito", f"Matriz aleatoria de {size}x{size} generada exitosamente.")
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    # Ingreso manual de la matriz
    def open_manual_input_window(self):
        try:
            size = int(self.entry_1.get())
            if size < 5 or size > 16:
                raise ValueError("El tamaño debe estar entre 5 y 16")
            
            input_window = tk.Toplevel(self.window)
            input_window.title("Ingreso Manual de la Matriz")
            
            entries = []
            for i in range(size):
                row_entries = []
                for j in range(size):
                    entry = ttk.Entry(input_window, width=5)
                    entry.grid(row=i, column=j, padx=2, pady=2)
                    row_entries.append(entry)
                entries.append(row_entries)
            
            ttk.Button(input_window, text="Aceptar", 
                       command=lambda: self.process_manual_input(entries, size, input_window)).grid(
                           row=size, column=0, columnspan=size, pady=10)
        
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    # Procesar la entrada manual
    def process_manual_input(self, entries, size, window):
        try:
            matrix = []
            for i in range(size):
                row = []
                for j in range(size):
                    value = int(entries[i][j].get())
                    if i == j and value != 0:
                        raise ValueError(f"El valor en la posición ({i+1},{j+1}) debe ser 0")
                    row.append(value)
                matrix.append(row)
            
            self.solver.distances = np.array(matrix)
            self.solver.n = size
            window.destroy()
            messagebox.showinfo("Éxito", f"Matriz de {size}x{size} ingresada exitosamente.")
            self.solver.visualize_interactive_graph()
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    # Ventana de resultados
    def solve_tsp(self):
        if self.solver.distances is None:
            messagebox.showerror("Error", "Por favor, genere una matriz antes de resolver el TSP.")
            return

        path, distance, execution_time, algorithm = self.solver.solve_tsp()
        
        result_window = tk.Toplevel(self.window)
        result_window.title("Resultados")
        result_window.geometry("600x150")

        ttk.Label(result_window, text=f"Algoritmo utilizado: {algorithm}").pack()
        ttk.Label(result_window, text=f"Tiempo de ejecución: {execution_time:.4f} segundos").pack()
        ttk.Label(result_window, text="Ruta óptima:").pack()
        ttk.Label(result_window, text=" -> ".join(str(city) for city in path + [path[0]])).pack()
        ttk.Label(result_window, text=f"Distancia total: {distance}").pack()

        self.solver.visualize_interactive_graph()
        self.solver.visualize_interactive_route(path, "Ruta Óptima usando {}".format(algorithm))

def main():
    window = tk.Tk()
    app = TSPGUI(window)
    window.resizable(False, False)
    window.mainloop()

if __name__ == "__main__":
    main()