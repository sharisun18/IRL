package IRLcode;

import burlap.behavior.singleagent.learnfromdemo.mlirl.differentiableplanners.DifferentiableVI;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import org.apache.commons.math3.linear.RealMatrix;


public interface DomainDistribution {

    SADomain sample();

    GridWorldState getState(SADomain domain);

    RealMatrix getPpi(Object policy, SADomain domain);

    RealMatrix[] getTransP();

    void launchExplorer(SADomain domain);

    void visualizePolicy(SADomain domain, Object policy, State initialState, DifferentiableVI dplanner);
}
