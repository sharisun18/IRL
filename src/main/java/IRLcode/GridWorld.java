package IRLcode;

import burlap.behavior.functionapproximation.dense.DenseStateFeatures;
import burlap.behavior.policy.BoltzmannQPolicy;
import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.policy.support.ActionProb;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.learnfromdemo.RewardValueProjection;
import burlap.behavior.singleagent.learnfromdemo.mlirl.MLIRL;
import burlap.behavior.singleagent.learnfromdemo.mlirl.MLIRLRequest;
import burlap.behavior.singleagent.learnfromdemo.mlirl.commonrfs.LinearStateDifferentiableRF;
import burlap.behavior.singleagent.learnfromdemo.mlirl.differentiableplanners.DifferentiableSparseSampling;
import burlap.behavior.singleagent.learnfromdemo.mlirl.differentiableplanners.DifferentiableVI;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.valuefunction.QProvider;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.debugtools.RandomFactory;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.StateGenerator;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.action.ActionUtils;
import burlap.mdp.core.oo.OODomain;
import burlap.mdp.core.oo.propositional.GroundedProp;
import burlap.mdp.core.oo.propositional.PropositionalFunction;
import burlap.mdp.core.oo.state.OOState;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.common.VisualActionObserver;
import burlap.mdp.singleagent.environment.EnvironmentOutcome;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.FullModel;
import burlap.mdp.singleagent.model.SampleModel;
import burlap.mdp.singleagent.model.TransitionProb;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.shell.visual.VisualExplorer;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;
import org.apache.commons.math3.linear.*;
import org.apache.commons.lang.math.RandomUtils;

import javax.xml.stream.Location;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.sql.Array;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.lang.System;


public class GridWorld implements DomainDistribution {

    int maxX;
    int maxY;
    int Nloc = 3;

    public GridWorld(int maxX, int maxY) {
        this.maxX = maxX;
        this.maxY = maxY;
    }

    @Override
    public SADomain sample() {

        java.util.Random random = new Random();
        GridLocation[] locations = new GridLocation[maxX * maxY];

        for (int i = 0; i < maxX; i++) {
            for (int j = 0; j < maxY; j++) {

                double r = random.nextGaussian();

                if (r < -1) { locations[5*i+j] = new GridLocation(i, j, 2, "loc"+Integer.toString(5*i+j)); } else
                if (r > 1 ) { locations[5*i+j] = new GridLocation(i, j, 1, "loc"+Integer.toString(5*i+j)); }
                else        { locations[5*i+j] = new GridLocation(i, j, 0, "loc"+Integer.toString(5*i+j)); }
            }
        }
        // Write list of locations to current environment
        GridWorldState s = new GridWorldState( new GridAgent(0, 0), locations );                // Default agent start location at (0, 0)

        SADomain domain = new SADomain();

        SampleModel gridModel = new GridModel(locations);
        domain.setModel(gridModel);

        return domain;
    }
    @Override
    public GridWorldState getState(SADomain domain) {
        return new GridWorldState( new GridAgent(0,0), ((GridModel)domain.getModel()).locations );
    }
    @Override
    public RealMatrix getPpi(Object policy, SADomain domain) {

        double[][]   Ppi    = new double[maxX*maxY][maxX*maxY];
        RealMatrix[] TransP = getTransP();

        for (int r = 0; r < maxX*maxY; r++) {
            for (int c = 0; c < maxX*maxY; c++) {
                for (int i = 0; i < 4; i++) {
                    Action action = (Action)domain.getAction(GridWorldDomain.ACTION_NORTH);
                    State currentState = new GridWorldState( new GridAgent(r, c), getLocations(domain) );
                    Ppi[r][c] += ((BoltzmannQPolicy)policy).actionProb(currentState, action) * TransP[i].getEntry(r,c);
                }
            }
        }
        return MatrixUtils.createRealMatrix(Ppi);
    }
    @Override
    public RealMatrix[] getTransP() {

        double[][][] TranP = new double[4][maxX*maxY][maxX*maxY];

        for (int r = 0; r < maxX*maxY; r++) {
            int startx = r / maxY; int starty = r % maxY;
            if (starty < maxY-1) { TranP[0][r][r + 1]    = 1.0; } else { TranP[0][r][r] = 1.0; }
            if (startx < maxX-1) { TranP[1][r][r + maxY] = 1.0; } else { TranP[1][r][r] = 1.0; }
            if (starty > 0)      { TranP[2][r][r - 1]    = 1.0; } else { TranP[2][r][r] = 1.0; }
            if (startx > 0)      { TranP[3][r][r - maxY] = 1.0; } else { TranP[3][r][r] = 1.0; }
        }
        RealMatrix[] TranPMtx = new RealMatrix[4];
        for (int i = 0; i < 4; i++) { TranPMtx[i] = MatrixUtils.createRealMatrix(TranP[i]); }

        return TranPMtx;
    }
    @Override
    public void launchExplorer(SADomain env) {

        GridWorldDomain gwd = new GridWorldDomain(maxX, maxY);
        gwd.setNumberOfLocationTypes(Nloc);

        State bs = new GridWorldState(new GridAgent(0,0), ((GridModel)env.getModel()).locations);
        StateGenerator sg = new LeftSideGen(5, bs);

        Visualizer v = GridWorldVisualizer.getVisualizer(gwd.getMap());

        SimulatedEnvironment e = new SimulatedEnvironment(env, sg);
        VisualExplorer exp = new VisualExplorer(env, e, v, 800, 800);
        exp.addKeyAction("w", GridWorldDomain.ACTION_NORTH, "");
        exp.addKeyAction("s", GridWorldDomain.ACTION_SOUTH, "");
        exp.addKeyAction("d", GridWorldDomain.ACTION_EAST,  "");
        exp.addKeyAction("a", GridWorldDomain.ACTION_WEST,  "");

        exp.initGUI();
    }
    @Override
    public void visualizePolicy(SADomain domain, Object policy, State initialState, DifferentiableVI dplanner) {

        SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();
        List<State> allStates = StateReachability.getReachableStates(initialState, domain, hashingFactory);

        ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(
                                         allStates, 5, 5, dplanner, (BoltzmannQPolicy)policy);
        gui.initGUI();
    }

    private class GridModel implements SampleModel {

        GridLocation[] locations;

        public GridModel(GridLocation[] locations) {
            this.locations = locations;
        }

        @Override
        public EnvironmentOutcome sample(State state, Action action) {

            GridAgent agent = ((GridWorldState)state).touchAgent();
            String actionName = action.actionName();

            State outcomeState = getOutcomeState(agent.x, agent.y, actionName);

            return new EnvironmentOutcome(state, action, outcomeState, 0, false);                          // TODO: r set to 0
        }

        public State getOutcomeState(int x, int y, String actionName) {

            double[][][] TransP = getTransP();




            List<String> actions = Arrays.asList(GridWorldDomain.ACTION_NORTH, GridWorldDomain.ACTION_EAST, GridWorldDomain.ACTION_SOUTH, GridWorldDomain.ACTION_WEST);
            int actionId = actions.indexOf(actionName);

            int outcomeStateId = arrayIndex(TransP[actionId][maxX*x+y], 1.0);
            int outcomeX = getFromId(outcomeStateId, 'x'); int outcomeY = getFromId(outcomeStateId, 'y');

            return new GridWorldState(new GridAgent(outcomeX, outcomeY), locations);
        }

        public int getFromId(int idInArray, char axis) {

            switch(axis) {
                case 'x': return idInArray / maxX;
                case 'y': return idInArray % maxX;
            }             return -1;
        }

        @Override
        public boolean terminal(State state) {
            return false;
        }

        /* getTransP
            1st layer: Action (n, e, s, w)
            2nd layer: Start state
            3rd layer: End state
        */
        public double[][][] getTransP() {

            double[][][] TranP;
            TranP = new double[4][maxX*maxY][maxX*maxY];

            // Inner square
            for (int r = 0; r < maxX*maxY; r++) {
                int startx = r / maxY; int starty = r % maxY;
                if (starty < maxY-1) { TranP[0][r][r + 1]    = 1.0; } else { TranP[0][r][r] = 1.0; }
                if (startx < maxX-1) { TranP[1][r][r + maxY] = 1.0; } else { TranP[1][r][r] = 1.0; }
                if (starty > 0)      { TranP[2][r][r - 1]    = 1.0; } else { TranP[2][r][r] = 1.0; }
                if (startx > 0)      { TranP[3][r][r - maxY] = 1.0; } else { TranP[3][r][r] = 1.0; }
            }
            return TranP;
        }

        public int arrayIndex(double[] array, double value) {
            for (int i = 0; i < array.length; i++) {
                if (array[i] == value) { return i; }
            }
            return -1;
        }
    }

    private static class LeftSideGen implements StateGenerator {

        protected int height;
        protected State sourceState;


        public LeftSideGen(int height, State sourceState){
            this.setParams(height, sourceState);
        }

        public void setParams(int height, State sourceState){
            this.height = height;
            this.sourceState = sourceState;
        }

        public State generateState() {

            GridWorldState s = (GridWorldState)this.sourceState.copy();

            int h = RandomFactory.getDefault().nextInt(this.height);
            s.touchAgent().y = h;

            return s;
        }
    }

    private GridLocation[] getLocations(SADomain domain) {
        return ((GridModel)domain.getModel()).locations;
    }

}


class GridAction implements Action {

    String actionName;

    public GridAction(String actionName) { this.actionName = actionName; }

    @Override
    public String actionName() { return actionName; }

    @Override
    public Action copy() { return this; }
}


/* Test codes

        GridWorld gridWorld = new GridWorld(5, 5);

        SADomain domain = gridWorld.sample();

        State  state  = new GridWorldState(new GridAgent(4,4), new GridLocation(4, 4, "loc0"));
        Action action = new GridAction(GridWorldDomain.ACTION_EAST);

        EnvironmentOutcome eo = domain.getModel().sample(state, action);

        System.out.println(eo.op.get("agent:x"));
        System.out.println(eo.op.get("agent:y"));

*/
