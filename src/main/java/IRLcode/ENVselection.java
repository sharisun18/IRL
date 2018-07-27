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
import burlap.mdp.singleagent.environment.EnvironmentOutcome;
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


public class ENVselection {

    private static int Nround = 2;

    private static int Nrf  = 1;
    private static int Nloc = 3;

    private static int maxX = 5;
    private static int maxY = 5;

    private static GridWorldDomain gwd;
    public static OOSADomain domain;

//    public DomainDistribution DD;
    public GridWorld DD = new GridWorld(5, 5);

    public GridRSampler SAMPLER = new GridRSampler(3, 5, 5);


    public ENVselection() {

        gwd = new GridWorldDomain(maxX, maxY);
        gwd.setNumberOfLocationTypes(Nloc);
        domain = gwd.generateDomain();
    }


    /* ------------------------------------------------- Sampling --------------------------------------------------- */


     /* sampleEnv
     *  For each grid in domain, sample from three locations using normal distribution
     *  locations type_2 is the most possible one (denoting paths)
     *  locations type_0 & type_1 denote pass & forbidden states (start & end states)
     */
    public SADomain[] sampleEnv(int Nenv) {

        SADomain[] ENV = new SADomain[Nenv];
        for (int k = 0; k < Nenv; k++) { ENV[k] = DD.sample(); }
        return ENV;
    }


    public LinearStateDifferentiableRF[] sampleR(int numRoundExecuted,
                                                 SADomain[] observedEs, int[][] observedTs) {              // TODO

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


    public void interactiveIRL() {

        SADomain[] observedEs = new SADomain[Nround];
        int[][]    observedTs = new int[Nround][3];

        int numRoundExecuted = 0;
        LinearStateDifferentiableRF[] Rs = sampleR( numRoundExecuted, observedEs, observedTs );

        while (numRoundExecuted < Nround) {

            SADomain newE = selectEnv(Rs, numRoundExecuted, observedEs, observedTs);

            DD.launchExplorer(newE);






            numRoundExecuted ++;
        }
    }


    public SADomain selectEnv(LinearStateDifferentiableRF[] Rs, int totalRound,
                                    SADomain[] observedEs, int[][] observedTs) {

        SADomain[] Es = sampleEnv(1);

        int maxValidRfOfEnv = 0; int bestEnv = 0;                                                                                                // TODO init env cannot be an arbitrary env

        for (int i = 0; i < Es.length; i++) {

            int maxNumValidRf = getMaxNumValidRf(Rs, Es, i);
            if (maxNumValidRf > maxValidRfOfEnv) { maxValidRfOfEnv = maxNumValidRf; bestEnv = i; }
        }
        return Es[bestEnv];
    }

    public int getMaxNumValidRf(LinearStateDifferentiableRF[] Rs, SADomain[] Es, int EnvId) {

        SADomain currentEnv = Es[EnvId];
        int maxNumValidRf = 0;

        for (int j = 0; j < Rs.length; j++) {
                                                                                                                        // TODO
            DifferentiableVI dplanner = getDPlanner(Rs[j]);                                                             // TODO
            BoltzmannQPolicy policy = dplanner.planFromState(DD.getState(currentEnv));

            RealMatrix Ppi = DD.getPpi(policy, currentEnv);

            int numValidRf = getNumValidRf(Rs, DD.getTransP(), Ppi);
            if (numValidRf > maxNumValidRf) { maxNumValidRf = numValidRf; }
        }
        return maxNumValidRf;
    }


    public int getNumValidRf(LinearStateDifferentiableRF[] Rs, RealMatrix[] TransP, RealMatrix Ppi) {

        int numValidRf = 0;

        for (LinearStateDifferentiableRF rf : Rs) {
            if ( SAMPLER.isValid(getRFvector(rf), TransP, Ppi) ) { numValidRf += 1; }                                 // TODO math problem
        }
        return numValidRf;
    }


    public static DifferentiableVI getDPlanner(LinearStateDifferentiableRF rf) {

        return new DifferentiableVI(domain, rf, 0.99, 10, new SimpleHashableStateFactory(), 0.01, 100);
//        return new ValueIteration(domain, 0.99, new SimpleHashableStateFactory(), 0.01, 100);
//        return new DifferentiableSparseSampling(domain, rf, 0.99, new SimpleHashableStateFactory(), 10, -1, 10);
    }


    public RealVector getRFvector(LinearStateDifferentiableRF rf) {

        double[] rf_ = new double[maxX*maxY];

        for (int i = 0; i < rf.numParameters(); i++) {
            rf_[i] = rf.getParameter(i);
        }
        return new ArrayRealVector(rf_);
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


    /**
     / --------------------------------------------------------------------------------------------------------------- /
     /                                                      Main                                                       /
     / --------------------------------------------------------------------------------------------------------------- /
     **/

    public static void main(String[] args) throws IOException {

        ENVselection ES = new ENVselection();
        ES.interactiveIRL();

    }

}