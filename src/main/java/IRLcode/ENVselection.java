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
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.FullModel;
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
import java.util.List;
import java.util.Random;
import java.lang.System;

import static com.sun.tools.classfile.AccessFlags.Kind.Field;
import static corejava.Format.printf;
import static jdk.nashorn.internal.objects.Global.println;
import static org.apache.commons.lang.math.RandomUtils.nextDouble;
import static org.apache.commons.math3.linear.MatrixUtils.createRealIdentityMatrix;
import static org.apache.commons.math3.linear.MatrixUtils.inverse;

public class ENVselection {

    public static int Nround = 2;

    public static int Nenv = 1;
    public static int Nrf  = 1;
    public static int Nloc = 3;

    public static int maxX = 5;
    public static int maxY = 5;

    public static GridWorldDomain gwd;
    public static OOSADomain domain;

    public static RealMatrix[] TransP;

//    public static String pathToEpisodes;  // TODO: set total rewards to be sampled

    // constructor
    public ENVselection() {

//        pathToEpisodes = "demos";

        gwd = new GridWorldDomain(maxX, maxY);
        gwd.setNumberOfLocationTypes(Nloc);
        domain = gwd.generateDomain();

        TransP = getTransitionProbs();
    }


    /* ------------------------------------------------- Sampling --------------------------------------------------- */


     /* sampleEnv
     *  For each grid in domain, sample from three locations using normal distribution
     *  locations type_2 is the most possible one (denoting paths)
     *  locations type_0 & type_1 denote pass & forbidden states (start & end states)
     */
    public static GridWorldState[] sampleEnv() {

        java.util.Random random = new Random();
        GridWorldState[] ENV    = new GridWorldState[Nenv];

        for (int k = 0; k < Nenv; k++) {
            // Store loc type at all domain locations / set of GridLocations
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
            ENV[k] = s;
        }
        return ENV;
    }


    public static LinearStateDifferentiableRF[] sampleR(int numRoundExecuted,
                                                        GridWorldState[] observedEs, int[][] observedTs) {              // TODO

        LinearStateDifferentiableRF[] Rs       = new LinearStateDifferentiableRF[Nrf];
        LocationFeatures              features = new LocationFeatures(domain, Nloc);

        for (int i = 0; i < Nrf; i++) {
            LinearStateDifferentiableRF rf = new LinearStateDifferentiableRF(features, Nloc);

            for (int j = 0; j < rf.numParameters(); j++) {
                rf.setParameter(j, RandomFactory.getMapped(0).nextDouble() * 2.0 - 1.0);
            }
            Rs[i] = rf;
        }
        return Rs;
    }


    /**
     / --------------------------------------------------------------------------------------------------------------- /
     /                                              Interactive Process                                                /
     / --------------------------------------------------------------------------------------------------------------- /
     **/


    public void interactiveIRL() throws FileNotFoundException {

        GridWorldState[] observedEs = new GridWorldState[Nround];
        int[][]          observedTs = new int[Nround][3]; // [Nround][{x, y, direc}]

        int numRoundExecuted = 0;
        LinearStateDifferentiableRF[] Rs = sampleR( numRoundExecuted, observedEs, observedTs );

        while (numRoundExecuted < Nround) {

            GridWorldState newE = selectEnv(Rs, numRoundExecuted, observedEs, observedTs);

            IRLExample ex = new IRLExample(newE);
            ex.launchExplorer();






            numRoundExecuted ++;
        }
    }


    public GridWorldState selectEnv(LinearStateDifferentiableRF[] Rs, int totalRound,
                                    GridWorldState[] observedEs, int[][] observedTs) {

        GridWorldState[] Es = sampleEnv();

        int maxValidRfOfEnv = 0; int bestEnv = 0;                                                                                                // TODO init env cannot be an arbitrary env

        for (int i = 0; i < Es.length; i++) {

            int maxNumValidRf = getMaxNumValidRf(Rs, Es, i);
            if (maxNumValidRf > maxValidRfOfEnv) { maxValidRfOfEnv = maxNumValidRf; bestEnv = i; }
        }
        return Es[bestEnv];
    }

    public int getMaxNumValidRf(LinearStateDifferentiableRF[] Rs, GridWorldState[] Es, int EnvId) {

        GridWorldState CurrentEnv = Es[EnvId];
        int maxNumValidRf = 0;

        for (int j = 0; j < Rs.length; j++) {
                                                                                                                        // TODO
            DifferentiableVI dplanner = getDPlanner(Rs[j]);                                                             // TODO
            BoltzmannQPolicy policy = dplanner.planFromState(CurrentEnv);

//            visualizePilicy(CurrentEnv, dplanner, policy);

            RealMatrix Ppi = getPpi(policy, Es[0].touchLocations());

            int numValidRf = getNumValidRf(Rs, TransP, Ppi);
            if (numValidRf > maxNumValidRf) { maxNumValidRf = numValidRf; }
        }
        return maxNumValidRf;
    }


    public int getNumValidRf(LinearStateDifferentiableRF[] Rs, RealMatrix[] TransP, RealMatrix Ppi) {

        int numValidRf = 0;
        IRLsampler Rsampler = new IRLsampler(Nloc, Nrf, maxX, maxY);
        for (LinearStateDifferentiableRF rf : Rs) {
            if ( Rsampler.isValid(getRFvector(rf), TransP, Ppi) ) { numValidRf += 1; }                                 // TODO math problem
        }
        return numValidRf;
    }


    public static DifferentiableVI getDPlanner(LinearStateDifferentiableRF rf) {

        return new DifferentiableVI(domain, rf, 0.99, 10, new SimpleHashableStateFactory(), 0.01, 100);

//        return new ValueIteration(domain, 0.99, new SimpleHashableStateFactory(), 0.01, 100);

//        return new DifferentiableSparseSampling(domain, rf, 0.99, new SimpleHashableStateFactory(),
//                                                10, -1, 10);
    }


    public RealMatrix getPpi(BoltzmannQPolicy policy, List<GridLocation> locations) {

        double[][] Ppi = new double[maxX*maxY][maxX*maxY];

        for (int r = 0; r < maxX*maxY; r++) {
            for (int c = 0; c < maxX*maxY; c++) {
                for (int i = 0; i < 4; i++) {
                    Action action = (Action)domain.getAction(GridWorldDomain.ACTION_NORTH);
                    State currentState = new GridWorldState( new GridAgent(r, c), locations);
                    Ppi[r][c] += policy.actionProb(currentState, action) * TransP[i].getEntry(r,c);
                }
            }
        }
        return MatrixUtils.createRealMatrix(Ppi);
    }


    public RealVector getRFvector(LinearStateDifferentiableRF rf) {

        double[] rf_ = new double[maxX*maxY];

        for (int i = 0; i < rf.numParameters(); i++) {
            rf_[i] = rf.getParameter(i);
        }
        return new ArrayRealVector(rf_);
    }


    public static void visualizePilicy(State initialState,
                                       DifferentiableVI dplanner, BoltzmannQPolicy P) {

        SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();
        List<State> allStates = StateReachability.getReachableStates(initialState, domain, hashingFactory);

        ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(
                allStates, 5, 5, dplanner, P);
        gui.initGUI();
    }

    /**
     * A state feature vector generator that create a binary feature vector where each element
     * indicates whether the agent is in a cell of a different type. All zeros indicates
     * that the agent is in an empty cell.
     */
    public static class LocationFeatures implements DenseStateFeatures {

        protected int numLocations;
        PropositionalFunction inLocationPF;


        public LocationFeatures(OODomain domain, int numLocations){
            this.numLocations = numLocations;
            this.inLocationPF = domain.propFunction(GridWorldDomain.PF_AT_LOCATION);
        }

        public LocationFeatures(int numLocations, PropositionalFunction inLocationPF) {
            this.numLocations = numLocations;
            this.inLocationPF = inLocationPF;
        }

        @Override
        public double[] features(State s) {

            double [] fv = new double[this.numLocations];

            int aL = this.getActiveLocationVal((OOState)s);
            if ( aL != -1 ) {
                fv[aL] = 1.;
            }

            return fv;
        }


        protected int getActiveLocationVal(OOState s){

            List<GroundedProp> gps = this.inLocationPF.allGroundings(s);
            for(GroundedProp gp : gps){
                if(gp.isTrue(s)){
                    GridLocation l = (GridLocation)s.object(gp.params[1]);
                    return l.type;
                }
            }

            return -1;
        }

        @Override
        public DenseStateFeatures copy() {
            return new IRLExample.LocationFeatures(numLocations, inLocationPF);
        }
    }


    public RealMatrix[] getTransitionProbs() {

        System.out.format("Getting transition probabilities\n\n");

        double[][][] TranP = new double[4][maxX*maxY][maxX*maxY];

        // Inner square
        for (int r = 0; r < maxX*maxY; r++) {
            int startx = r / maxY; int starty = r % maxY;
            if (starty < maxY-1) { TranP[0][r][r + 1]    = 1.0; } else { TranP[0][r][r] = 1.0; }
            if (startx < maxX-1) { TranP[1][r][r + maxY] = 1.0; } else { TranP[1][r][r] = 1.0; }
            if (starty > 0)      { TranP[2][r][r - 1]    = 1.0; } else { TranP[2][r][r] = 1.0; }
            if (startx > 0)      { TranP[3][r][r - maxY] = 1.0; } else { TranP[3][r][r] = 1.0; }
        }

        // Print out all transition probs for all 4 directions                                                  // TODO
//        for (int c = 0; c < 4; c++) {
//            System.out.format("DIRECTION: %d\n\n", c);
//
//            for (int i = 0; i < maxX * maxY; i++) {
//                for (int j = 0; j < maxX * maxY; j++) {
//                    System.out.format("%.2f ", TranP[c][i][j]);
//                }
//                System.out.format("\n");
//            }
//            System.out.format("\n");
//        }


        // Stored as Apache matrix
        RealMatrix[] TranP_M = new RealMatrix[4];
        for (int i = 0; i < 4; i++) { TranP_M[i] = MatrixUtils.createRealMatrix(TranP[i]); }

        return TranP_M;
    }

//    private static double[] parseActionProb(List<ActionProb> actionProbs) {
//
//        double[] aProbs = new double[4];
//
//        aProbs[0] = Double.parseDouble( actionProbs.get(0).toString().split(": ")[0] );
//        aProbs[1] = Double.parseDouble( actionProbs.get(2).toString().split(": ")[0] );
//        aProbs[2] = Double.parseDouble( actionProbs.get(1).toString().split(": ")[0] );
//        aProbs[3] = Double.parseDouble( actionProbs.get(3).toString().split(": ")[0] );
//
//        return aProbs;
//    }


//    private static double[] addActionProb(double[] pi, int x, int y, double[] aProbs) {
//
//        int id = 4 * (maxX*x + y); // start index
//
//        for (int i = 0; i < 4; i++) { pi[id+i] = aProbs[i]; }
//
////        System.arraycopy(aProbs, 0, pi, id, 4);
//
//        return pi;
//    }


//    public static double[] getPiVector(BoltzmannQPolicy policy, List<GridLocation> locations) {
//
//        double[] pi = new double[4*maxX*maxY];
//
//        for (int x = 0; x < maxX; x++) {
//            for (int y = 0; y < maxY; y++) {
//
//                GridWorldState curState = new GridWorldState( new GridAgent(x, y), locations );
//                List<ActionProb> actionProbs = policy.policyDistribution(curState);
//                double[] aProbs = parseActionProb(actionProbs);
//                pi = addActionProb(pi, x, y, aProbs);
//            }
//        }
//        return pi;
//    }



    /**
     / --------------------------------------------------------------------------------------------------------------- /
     /                                                      Main                                                       /
     / --------------------------------------------------------------------------------------------------------------- /
     **/

    public static void main(String[] args) throws IOException {

//        String pathToEpisodes = "demos/4demo0.episode";

        ENVselection     ES = new ENVselection();
        ES.interactiveIRL();



//        GridWorldState[] Es = sampleEnv();

        //  main test

        /*
        GridWorldState[] Es = sampleEnv();


        // domain
        GridWorldDomain gwd = new GridWorldDomain(maxX, maxY);
        gwd.setNumberOfLocationTypes(Nloc);
        OOSADomain domain = gwd.generateDomain();


        // reward function
        IRLExample.LocationFeatures features = new IRLExample.LocationFeatures(domain, Nloc);
        LinearStateDifferentiableRF rf = new LinearStateDifferentiableRF(features, Nloc);
        for (int j = 0; j < rf.numParameters(); j++) {
            rf.setParameter(j, RandomFactory.getMapped(0).nextDouble() * 2.0 - 1.0);
        }


        // policy
        DifferentiableSparseSampling dplanner = new DifferentiableSparseSampling(domain, rf, 0.99,
                new SimpleHashableStateFactory(),
                10, -1, 10);
        GreedyQPolicy qp = new GreedyQPolicy(dplanner);


        // policy vis
        State initialState = Es[0];
        SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();
        List<State> allStates = StateReachability.getReachableStates(initialState, domain, hashingFactory);



        ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(
                                         allStates, 5, 5, dplanner, qp);


        // test
        GridWorldState curState = new GridWorldState( new GridAgent(0, 0), Es[0].touchLocations() );
        List<ActionProb> actionProbs = qp.policyDistribution(curState);
        for (ActionProb actionProb : actionProbs) { System.out.format("%s\n", actionProb.toString()); }

        curState = new GridWorldState( new GridAgent(0, 1), Es[0].touchLocations() );
        actionProbs = qp.policyDistribution(curState);
        for (ActionProb actionProb : actionProbs) { System.out.format("%s\n", actionProb.toString()); }

        curState = new GridWorldState( new GridAgent(0, 2), Es[0].touchLocations() );
        actionProbs = qp.policyDistribution(curState);
        for (ActionProb actionProb : actionProbs) { System.out.format("%s\n", actionProb.toString()); }

        curState = new GridWorldState( new GridAgent(0, 3), Es[0].touchLocations() );
        actionProbs = qp.policyDistribution(curState);
        for (ActionProb actionProb : actionProbs) { System.out.format("%s\n", actionProb.toString()); }


        gui.initGUI();

        */

        /* For each environment,
         *     1. Get a set of reward functions from episodes for this environment (for now just one RF),
         *     2. Get policy for each ENV-RF pair.
         */
//        for (GridWorldState env : Es) {
//            IRLExample EX = new IRLExample(env);
//
//            EX.launchExplorer(); //choose this to record demonstrations
//        }

    }

}