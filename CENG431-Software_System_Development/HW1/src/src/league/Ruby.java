package src.league;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import src.user.User;

public class Ruby extends League {
    private List<User> leaderBoard;
    private String leagueName;

    public Ruby(String leagueName) {
    	super(leagueName);
        this.leagueName = leagueName;
        this.leaderBoard = new ArrayList<>();
    }



    @Override
    public void sortLeaderBoard() {
        this.leaderBoard.sort(Comparator.comparingInt((User u) -> u.getTotalPoints()));
    }

    @Override
    public void promoteLeaderBoard(ILeague upperLeague) {

    }
}
