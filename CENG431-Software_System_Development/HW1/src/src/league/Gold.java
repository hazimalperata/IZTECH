package src.league;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import src.user.User;

public class Gold extends League {
	private List<User> leaderBoard;
	private  String leagueName;

	public Gold(String leagueName) {
    	super(leagueName);
		this.leagueName = leagueName;
		this.leaderBoard = new ArrayList<>();
	}

	@Override
	public List<User> popPromoteableUsers(int count) {
		List<User> resultList = new ArrayList<>();
		for (int i = 0; i < count; i++) {
			if (leaderBoard.size() > i && leaderBoard.get(i) != null && leaderBoard.get(i).getNumberOfDaysStreak() >= 7)
				resultList.add(this.leaderBoard.remove(i));
		}
		return resultList;
	}

	@Override
	public void promoteLeaderBoard(ILeague upperLeague) {
		upperLeague.appendUsers(this.popPromoteableUsers(5));
	}
}
