package src.league;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import src.user.User;

public abstract class League implements ILeague {
    private List<User> leaderBoard;
    private String leagueName;

//    public League() {
//        this("");
//    }

    public League(String leagueName) {
        this.leagueName = leagueName;
        this.leaderBoard = new ArrayList<>();
    }

    @Override
    public String getLeagueName() {
        return leagueName;
    }

    @Override
    public List<User> getLeaderBoard() {
        return leaderBoard;
    }


    @Override
    public int getUserCount() {
        return this.leaderBoard.size();
    }

    @Override
    public List<User> popPromoteableUsers(int count) {
        List<User> resultList = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            if (leaderBoard.size() > i && leaderBoard.get(i) != null)
                resultList.add(this.leaderBoard.remove(i));
        }
        return resultList;
    }

    @Override
    public void appendUser(User newUser) {
        this.leaderBoard.add(newUser);
    }

    @Override
    public void appendUsers(List<User> newUsers) {
        this.leaderBoard.addAll(newUsers);
    }

    @Override
    public void sortLeaderBoard() {
        this.leaderBoard.sort((u1, u2) -> {
            if (u1.getTotalPoints() == u2.getTotalPoints()) {
                return u1.getNumberOfDaysStreak() - u2.getNumberOfDaysStreak();
            } else {
                return u1.getTotalPoints() - u2.getTotalPoints();
            }
        });
    }

    @Override
    public List<User> getTopUsers(int count) {
        List<User> resultList = new ArrayList<>();
        for (int i = 0; i < count; i++) {
        	if (this.leaderBoard.size() > i) {
        		 resultList.add(this.leaderBoard.get(i));
        	} else {
        		break;
        	}
        }
        return resultList;
    }
}
