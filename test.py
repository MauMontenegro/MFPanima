import numpy as np
from manim import *
import networkx as nx
import json
import math

class AnimateMFP(Scene):
    def construct(self):

        # Screen parameters
        height =8
        width = 14.2
        rows=10
        columns=20
        x_step = width/columns
        y_step = height/rows

        # MFP Parameters
        PROTECTED = BLUE
        SAVED = GREEN
        BURNED = RED
        labeled = set()
        fire = set()

        def burnVertices(fire_t):
            neighbors = []
            for fire_vertex in fire_t:
                neighbors = neighbors + list(G.neighbors(fire_vertex))
            self.play(*[g.vertices[e].animate.set_color(BURNED) for e in neighbors])
            self.wait(1)
            fire_t.clear()
            fire_t |= set(neighbors)

        # Load Instance
        G,G_dist,pos,starting_fire,N = self.loadPaperInstance()
        # Adding fire roots to initial fire vertices
        for fire_root in starting_fire:
            fire.add(fire_root)

        intG = [[float(f'{ele:.2f}') for ele in sub] for sub in G_dist]
        D= Matrix(intG)
        G_label = MarkupText(f'<span fgcolor="{GREEN}"> &lt; G </span>',color=WHITE).scale(.5).move_to([2.5*x_step , 3.5 * y_step, 0])
        F_label = MarkupText(f'<span fgcolor="{RED}">F</span>',color=WHITE).scale(.5).move_to([3.25*x_step, 3.5 * y_step, 0])
        a_label = MarkupText(f'<span fgcolor="{BLUE}">a</span>',color=WHITE).scale(0.5).move_to([3.75* x_step, 3.5 * y_step, 0])
        Ts_label = MarkupText(f'<span fgcolor="{WHITE}">Ts</span>',color=WHITE).scale(0.5).move_to([4.5* x_step, 3.5 * y_step, 0])
        f_label = MarkupText(f'<span fgcolor="{WHITE}"> f &gt; </span>', color=WHITE).scale(0.5).move_to([5*x_step, 3.5 * y_step, 0])
        D.scale(0.3).move_to([-4.5*x_step,-2.5*y_step,0])

        # Introduction
        Intro=[]
        Title = Tex("Moving Firefighter Problem").move_to([-4.5*x_step,4*y_step,0])
        Intro_1 = MarkupText(f'A MFP can be modeled by a <span fgcolor="{GREEN}"> Graph (G=(V,E))</span>',color=WHITE).scale(.3).move_to([-4.5*x_step,3*y_step,0])
        Intro_2 = MarkupText(f'A subset <span fgcolor="{RED}"> F</span> of vertices where the source of the spreading starts',color=WHITE)\
            .scale(.3).move_to([-4.5*x_step,2.5*y_step,0])
        Intro_3 = MarkupText(f'An initial depot position <span fgcolor="{BLUE}"> a</span> where the firefighter is deployed',color=WHITE)\
            .scale(.3).move_to([-4.5*x_step,2*y_step,0])
        Intro_4 =  MarkupText(f'A <b>T</b> function which gives distances between all vertices of G (including firefighter)',color=WHITE)\
            .scale(.3).move_to([-4.5*x_step,1.5*y_step,0])
        Intro_5 = MarkupText(f'A spreading ratio function <b>f</b>',color=WHITE).scale(.3).move_to([-4.5 * x_step, 1 * y_step, 0])
        Intro.extend([Intro_1,Intro_2,Intro_3,Intro_4,Intro_5])
        Intro_group= VGroup(*Intro)

        # Create Graph Mobject
        g = Graph(list(G.nodes), list(G.edges), layout=pos, labels=True,
                  vertex_config={'radius': 0.20}) \
            .scale(1).move_to([0,0,0])

        # Construct TimeLine
        timeline=[]
        timeline_0 = Line(start=([3*x_step,-2.5*y_step,0]),end=([7*x_step,-2.5*y_step,0]))
        timeline_1 = Line(start=([3*x_step,-2.7*y_step,0]),end=([3*x_step,-2.3*y_step,0]))
        timeline_2 = Line(start=([4*x_step, -2.7*y_step, 0]), end=([4*x_step, -2.3*y_step, 0]))
        timeline_3 = Line(start=([5*x_step, -2.7*y_step, 0]), end=([5*x_step, -2.3*y_step, 0]))
        timeline_4 = Line(start=([6*x_step, -2.7*y_step, 0]), end=([6*x_step, -2.3*y_step, 0]))
        timeline_5 = Line(start=([7*x_step, -2.7*y_step, 0]), end=([7*x_step, -2.3*y_step, 0]))
        t_init_pos=np.array([3*x_step,-2.5*y_step,0])
        timeline.extend([timeline_0,timeline_1,timeline_2,timeline_3,timeline_4,timeline_5])
        time_group=VGroup(*timeline)

        self.play(FadeIn(Title))
        self.play(Write(Intro_1))
        self.play(FadeIn(G_label))
        self.wait()
        self.play(Create(g), run_time=2)
        self.play(g.animate.move_to([4, 0, 0]).scale(0.9))
        self.play(Circumscribe(g))
        self.play(Write(Intro_2))
        self.play(FadeIn(F_label))
        # Position Initial Fire
        self.play(*[g.vertices[e].animate.set_color(BURNED) for e in fire])
        self.play(*[Flash(g.vertices[i],color=BURNED,flash_radius=0.09) for i in fire])
        self.play(Write(Intro_3))
        self.play(FadeIn(a_label))
        firefighter = (Circle(fill_color=PROTECTED, fill_opacity=1, stroke_color=PROTECTED)
                       .move_to(g.vertices[N]).scale(.18))
        self.play(FadeIn(firefighter))
        self.play(Flash(g.vertices[N]))
        self.wait()
        self.play(Write(Intro_4))
        self.play(FadeIn(D))
        self.play(Transform(D,Ts_label))
        self.play(Write(Intro_5))
        self.play(FadeIn(f_label))
        self.play(FadeIn(time_group))

        # Animated Timeline
        save_state=[g[v].save_state() for v in g.vertices]
        arrow = Arrow(start=t_init_pos+np.array([0,-1,0]), end=t_init_pos, color=GOLD)
        arrow_state=arrow.save_state()
        self.add(arrow)
        self.wait()
        fratetx = Tex("Fire Rate:").scale(0.5).move_to(t_init_pos+[0,0.5*y_step,0])
        self.add(fratetx)
        self.wait(1)
        fire_t = set()
        for f in range(1,3):
            fire_t |= fire
            self.play(Transform(fratetx,Tex("Fire rate:"+str(f)).scale(0.5).move_to(t_init_pos+[0,0.5*y_step,0])))
            for i in range(1,5):
                self.play(arrow.animate.shift(RIGHT*x_step))
                if i%f==0:
                    burnVertices(fire_t)
            fire_t.clear()
            self.play(*[vertex.animate.restore() for vertex in save_state])
            self.play(arrow.animate.restore())



        def defenseStrategy(starting_fire,firefighter,fire,agent_pos,neighbors,unexplored_neighbors):
            update = 1  # Rate of fire spread
            travel_time = 0       # time
            s = 0       # Solution index
            flag = 0
            while unexplored_neighbors:
                if flag==0:
                    travel_time += G_dist[agent_pos][solution[s]]
                print(travel_time)
                print((solution[s]))
                if travel_time <= update:
                    # Animate movement of firefighter
                    self.play(AnimationGroup(
                        firefighter.animate.move_to(g.vertices[solution[s]]),
                        g.vertices[solution[s]].animate.set_color(PROTECTED)))
                    labeled.add(solution[s])
                    self.wait()
                    # Move to next index Solution
                    agent_pos = solution[s]
                    s+=1
                    flag=0
                    if len(solution) <= s:
                        break

                # Enter if travel time is greater than update (Time to burn!)
                while (travel_time > update):
                    travel_time= travel_time - update
                    print(travel_time)
                    for fire_vertex in fire:
                        neighbors = neighbors + list(G.neighbors(fire_vertex))
                    for fire_root in fire:
                        labeled.add(fire_root)
                    unexplored_neighbors=[]
                    unexplored_neighbors = [w for w in neighbors if w not in labeled]

                    if unexplored_neighbors:
                        self.play(*[g.edges[e].animate.set_color(BURNED) for e in G.edges if
                                    (e[0] in fire and e[1] in unexplored_neighbors) or (e[1] in fire and e[0] in unexplored_neighbors)])
                        #print(unexplored_neighbors)
                        self.play(*[g.vertices[e].animate.set_color(BURNED) for e in unexplored_neighbors])
                        self.play(*[Flash(g.vertices[i],color=BURNED,flash_radius=0.09) for i in unexplored_neighbors])
                        labeled.add(tuple(unexplored_neighbors))
                    fire.clear()
                    fire |= set(unexplored_neighbors)
                    print(unexplored_neighbors)
                    neighbors=[]
                    flag=1


        solution = [5,6,7]
        agent_pos= N
        # Add initial unlabeled vertices
        neighbors=[]
        for fire_vertex in fire:
            neighbors = neighbors + list(G.neighbors(fire_vertex))
        for fire_root in fire:
            labeled.add(fire_root)
        unexplored_neighbors = [w for w in neighbors if w not in labeled]
        neighbors=[]

        self.play(FadeOut(Intro_group))
        solution_label = MarkupText(f'A feasible solution for <b>MFP</b> is a sequence of vertices: <span fgcolor="{BLUE}"> S </span>', color=WHITE).scale(0.3).move_to(
            [-3.5*x_step, 2.5 * y_step, 0])
        Solution = MarkupText(f'<span fgcolor="{BLUE}"> S=(a,u_1,...,u_l) </span>', color=WHITE).scale(0.3).move_to(solution_label.get_center()+[0,-1*y_step,0])
        solution_label_2=MarkupText("Such that satisfy next equation for i=1,2,...,l:").scale(0.3).move_to(Solution.get_center()+[0,-1*y_step,0])
        solution_label_3=MathTex(r"\tau(a,u_1) + \sum_{k=1}^{i-1} \tau(u_k,u_{k+1}) \le \beta_{\mathcal{S}_i}").scale(0.5).move_to(solution_label_2.get_center()+[0,-1*y_step,0])
        self.play(Write(solution_label))
        self.wait(1)
        self.play(Write(Solution))
        self.wait(1)
        self.play(Write(solution_label_2))
        self.wait(1)
        self.play(Write(solution_label_3))
        self.wait(1)
        #defenseStrategy(starting_fire,firefighter,fire,agent_pos,neighbors,unexplored_neighbors)

    def MoveScalingGraph(self,mob):
        mob.move_to([4, 0, 0])
        mob.scale(.5)
        return mob

    def loadInstance(self):
        # Load Tree Instance
        T = nx.read_adjlist("instance/MFF_Tree.adjlist")
        print(T)
        mapping = {}
        for node in T.nodes:
            mapping[node] = int(node)
        T = nx.relabel_nodes(T, mapping)

        # Distance Matrix
        T_Ad_Sym = np.load("instance/FDM_MFFP.npy")

        # Position of vertices
        layout = open("instance/layout_MFF.json")
        pos = {}
        pos_ = json.load(layout)

        for position in pos_:
            pos_[position].append(0)
            pos[int(position)] = pos_[position]

        # Get Instance Parameters
        p = open("instance/instance_info.json")
        parameters = json.load(p)
        N = parameters["N"]
        seed = parameters["seed"]
        scale = parameters["scale"]
        starting_fire=[]
        starting_fire.append(parameters["start_fire"])
        starting_fire.append(11)
        tree_height = parameters["tree_height"]
        #T = nx.bfs_tree(T, starting_fire)
        T.add_node(N)
        return T,T_Ad_Sym,pos,starting_fire,N

    def loadPaperInstance(self):
        scale=3.33
        node_pos = {}
        node_pos[0] = [0.12*scale, 0.278*scale]
        node_pos[1] = [0.17*scale, 0.7*scale]
        node_pos[2] = [0.37*scale, 0.06*scale]
        node_pos[3] = [0.27*scale, 0.6*scale]
        node_pos[4] = [0.31*scale, 0.78*scale]
        node_pos[5] = [0.69*scale, 0.5*scale]
        node_pos[6] = [0.54*scale, 0.71*scale]
        node_pos[7] = [0.71*scale, 0.89*scale]
        node_pos[8] = [0.87*scale, 0.16*scale]
        node_pos[9] = [0.92*scale, 0.8*scale]

        edges = [(0, 2), (1, 3), (2, 3), (2, 5), (3, 5), (3, 4), (3, 6), (4, 7), (5, 6), (5, 8), (6, 9)]
        burnt_nodes = [0,1]
        fighter_pos= [0.58*scale,0.18*scale]
        n=10

        G=nx.Graph()
        G.add_nodes_from(node_pos.keys())
        G.add_edges_from(edges)

        T_Ad = np.zeros((n + 1, n + 1))
        for row in range(0, n):
            for column in range(row, n):
                if row == column:
                    T_Ad[row][column] = 0
                else:
                    x_1 = node_pos[row][0]
                    x_2 = node_pos[column][0]
                    y_1 = node_pos[row][1]
                    y_2 = node_pos[column][1]
                    dist = np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)
                    T_Ad[row][column] = dist

        for node in range(0, n):
            x_1 = node_pos[node][0]
            x_2 = fighter_pos[0]
            y_1 = node_pos[node][1]
            y_2 = fighter_pos[1]
            dist = np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)
            T_Ad[node][n] = dist
        T_Ad_Sym = np.triu(T_Ad) + np.tril(T_Ad.T)
        G.add_node(n)
        node_pos[n] = fighter_pos

        # Position of vertices
        pos = {}
        for position in node_pos:
            node_pos[position].append(0)
            pos[int(position)] = node_pos[position]

        print(G.nodes)
        print(G.edges)
        print(node_pos)
        return G,T_Ad_Sym,pos,burnt_nodes,n