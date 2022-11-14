from manim import *
import networkx as nx
import json

class AnimateMFP(Scene):
    def construct(self):
        PROTECTED = BLUE
        SAVED = GREEN
        BURNED = RED
        labeled= set()
        fire = set()

        # Load Instance
        G,G_dist,pos,starting_fire,N = self.loadInstance()

        # Introduction
        t1= Tex("Moving Firefighter Problem").move_to([-4,3,0])
        p1=Tex("An MFP is defined as a 6-tuple: /< G,F,a,T,f,ts /> where").move_to(t1).move_to(DOWN*1).scale(.5)
        p2=MathTex("G=V,E")
        self.play(FadeIn(t1))
        #self.play(FadeIn(p1))
        # Create Graph Mobject
        g = Graph(G.nodes,G.edges,layout=pos,labels=True).scale(0.5).move_to([4,0,0])
        #firefighter.move_to([4, 0, 0])
        self.play(Create(g), run_time=2)
        self.wait()
        # Position Graph
        #self.play(ApplyFunction(self.MoveScalingGraph,g))
        self.add(g)
        self.wait()
        # Animate initial Fire And Agent
        #self.play(g.vertices[starting_fire].animate.set_color(RED))

        def defenseStrategy(starting_fire,firefighter,fire,agent_pos):
            update = 1
            neighbors=[]
            for solution_vertex in solution:
                #Movement of firefighter
                self.play(AnimationGroup(
                    firefighter.animate.move_to(g.vertices[solution_vertex]),
                    g.vertices[solution_vertex].animate.set_color(PROTECTED)))
                labeled.add(solution_vertex)
                self.wait()
                # Get travel time of distance matrix
                travel_time = G_dist[agent_pos][solution_vertex]
                while (travel_time > update):
                    print(travel_time)
                    travel_time -= update
                    for fire_vertex in fire:
                        neighbors = neighbors + list(G.neighbors(fire_vertex))
                    for fire_root in fire:
                        labeled.add(fire_root)
                    unexplored_neighbors = [w for w in neighbors if w not in labeled]
                    print(G.edges)
                    print(fire)
                    print(unexplored_neighbors)
                    if unexplored_neighbors:
                        self.play(*[g.edges[e].animate.set_color(BURNED) for e in G.edges if
                                    (e[0] in fire and e[1] in unexplored_neighbors) or (e[1] in fire and e[0] in unexplored_neighbors)])
                        #print(unexplored_neighbors)
                        self.play(*[g.vertices[e].animate.set_color(BURNED) for e in unexplored_neighbors])
                        self.play(*[Flash(g.vertices[i],color=BURNED,flash_radius=0.05) for i in unexplored_neighbors])
                        labeled.add(tuple(unexplored_neighbors))
                    fire.clear()
                    fire |= set(unexplored_neighbors)
                    unexplored_neighbors = []
                    neighbors=[]
                agent_pos = solution_vertex

        for fire_root in starting_fire:
            fire.add(fire_root)

        # Position Initial Fires and Agent
        self.play(*[g.vertices[e].animate.set_color(BURNED) for e in fire])
        firefighter = (Circle(fill_color=PROTECTED, fill_opacity=1, stroke_color=PROTECTED)
                       .move_to(g.vertices[N]).scale(.15))
        self.add(firefighter)
        self.wait()

        solution = [17]
        agent_pos= N
        defenseStrategy(starting_fire,firefighter,fire,agent_pos)

    def MoveScalingGraph(self,mob):
        mob.move_to([4, 0, 0])
        mob.scale(.5)
        return mob

    def loadInstance(self):
        # Load Tree Instance
        T = nx.read_adjlist("instance/MFF_Tree.adjlist")
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