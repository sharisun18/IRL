package IRLcode;

import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.ArrowActionGlyph;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.LandmarkColorBlendInterpolation;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.PolicyGlyphPainter2D;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.StateValuePainter2D;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.learning.tdmethods.SarsaLam;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.deterministic.DeterministicPlanner;
import burlap.behavior.singleagent.planning.deterministic.informed.Heuristic;
import burlap.behavior.singleagent.planning.deterministic.informed.astar.AStar;
import burlap.behavior.singleagent.planning.deterministic.uninformed.bfs.BFS;
import burlap.behavior.singleagent.planning.deterministic.uninformed.dfs.DFS;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.valuefunction.QFunction;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.stateconditiontest.StateConditionTest;
import burlap.mdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.state.State;
import burlap.mdp.core.state.vardomain.VariableDomain;
import burlap.mdp.singleagent.common.GoalBasedRF;
import burlap.mdp.singleagent.common.VisualActionObserver;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.FactoredModel;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;

import java.awt.*;
import java.util.List;

public class BasicBehavior {

    GridWorldDomain gwdg;
    OOSADomain domain;
    TerminalFunction tf;
    StateConditionTest goalCondition;
    State initialState;
    HashableStateFactory hashingFactory;
    SimulatedEnvironment env;


    public BasicBehavior() {
        gwdg = new GridWorldDomain(11, 11);
        gwdg.setMapToFourRooms();                               // A default setting of the package

        tf = new GridWorldTerminalFunction(10, 10);             // terminal location
        gwdg.setTf(tf);
        goalCondition = new TFGoalCondition(tf);

        domain = gwdg.generateDomain();

        initialState   = new GridWorldState(new GridAgent(0, 0), new GridLocation(10, 10, "loc0"));
        hashingFactory = new SimpleHashableStateFactory();      // TODO

        env = new SimulatedEnvironment(domain, initialState);

        //VisualActionObserver observer = new VisualActionObserver(domain,
        //	GridWorldVisualizer.getVisualizer(gwdg.getMap()));
        //observer.initGUI();
        //env.addObservers(observer);
    }

    // ------------------------------- visualizer --------------------------------

//    public void visualize(String outputpath){
//        Visualizer v = GridWorldVisualizer.getVisualizer(gwdg.getMap());
//        new EpisodeSequenceVisualizer(v, domain, outputpath);
//    }

    // ------------------------------- Q-Learning --------------------------------

    public void qLearningExample(String outputPath){

        LearningAgent agent = new QLearning(domain, 0.99, hashingFactory, 0., 1.);

        //run learning for 50 episodes
        for(int i = 0; i < 50; i++){
            Episode e = agent.runLearningEpisode(env);

            e.write(outputPath + "ql_" + i);
            System.out.println(i + ": " + e.maxTimeStep());

            //reset environment for next learning episode
            env.resetEnvironment();
        }

    }



    // ================================== MAIN ===================================

    public static void main(String[] args) {

        BasicBehavior IRLexample = new BasicBehavior();
        String outputPath = "outputs/";                         //directory to record results

        // Call planning and learning algorithms here
        IRLexample.qLearningExample(outputPath);



        // Run the visualizer
//        example.visualize(outputPath);

    }


}




