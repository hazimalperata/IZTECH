package src.league;

import java.util.ArrayList;
import java.util.List;

import src.user.User;

public class Sapphire extends League {
    private List<User> leaderBoard;
    private static String leagueName;

    public Sapphire(String leagueName) {
    	super(leagueName);
        this.leagueName = leagueName;
        this.leaderBoard = new ArrayList<>();
    }

    @Override
    public List<User> popPromoteableUsers(int count) {
        List<User> resultList = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            if (leaderBoard.size() > i) {
                User selectedUser = leaderBoard.get(i);
                if (selectedUser != null && selectedUser.getNumberOfDaysStreak() >= 30 && (selectedUser.getTotalPoints() >= 5000 || selectedUser.getCurrentUnit().getUnitNumber() >= 10))
                    resultList.add(this.leaderBoard.remove(i));
            }
        }
        return resultList;
    }

    @Override
    public void promoteLeaderBoard(ILeague upperLeague) {
        upperLeague.appendUsers(this.popPromoteableUsers(this.getUserCount()));
    }
}
