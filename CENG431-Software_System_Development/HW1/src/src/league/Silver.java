package src.league;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import src.user.User;

public class Silver extends League {
    private List<User> leaderBoard;
    private String leagueName;

    public Silver(String leagueName) {
    	super(leagueName);
        this.leagueName = leagueName;
        this.leaderBoard = new ArrayList<>();
    }

    @Override
    public void promoteLeaderBoard(ILeague upperLeague) {
        upperLeague.appendUsers(this.popPromoteableUsers(10));
    }

}
