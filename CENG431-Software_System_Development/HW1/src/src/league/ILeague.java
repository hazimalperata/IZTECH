package src.league;

import java.util.List;

import src.user.User;

public interface ILeague {
    List<User> getLeaderBoard();

    int getUserCount();

    String getLeagueName();

    void appendUser(User newUser);

    void appendUsers(List<User> newUsers);

    void sortLeaderBoard();

    List<User> getTopUsers(int count);

    List<User> popPromoteableUsers(int count);

    void promoteLeaderBoard(ILeague upperLeague);

}
