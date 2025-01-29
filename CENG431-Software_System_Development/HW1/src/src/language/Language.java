package src.language;

import src.league.ILeague;
import src.league.League;
import src.quiz.Quiz;
import src.unit.Unit;
import utils.Rand;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Language {
    private final int MAX_UNITS_NUMBER = 100;
    private final int MIN_UNITS_NUMBER = 60;

    private List<Unit> units;
    private String name;
    private List<ILeague> leagues;


    public Language(String name, List<ILeague> leagues) {
        this.name = name;
        this.leagues = leagues;
        createLanguage();
    }

    public Language(String name, List<Unit> units, List<ILeague> leagues) {
        this.name = name;
        this.units = units;
        this.leagues = leagues;
    }

    private void createLanguage() {
        Rand rand = new Rand();
        this.units = new ArrayList<>();
        int count = rand.getInt(MIN_UNITS_NUMBER, MAX_UNITS_NUMBER);
        for (int i = 0; i < count; i++) {
            Unit unit = new Unit("Unit %s".formatted(i + 1));
            units.add(unit);
        }
    }

    public void handleLeagues() {
        for (int i = 0; i < this.leagues.size() - 1; i++) {
            this.leagues.get(i).sortLeaderBoard();
            this.leagues.get(i).promoteLeaderBoard(this.leagues.get(i + 1));
        }
    }

    public ILeague getSilverLeague() {
        return this.leagues.get(1);
    }

    public ILeague getBronzeLeague() {
        return this.leagues.get(0);
    }

    public String getName() {
        return name;
    }

    public List<Unit> getUnits() {
        return new ArrayList<Unit>(units);
    }

    public Unit getUnit(int i) {
        return units.get(i);
    }

    public int getNumberOfUnits() {
        return this.units.size();
    }

    public int getNumberOfTotalQuizzes() {
        int quizNum = 0;
        for (int i = 0; i < units.size(); i++) {
            quizNum += getUnit(i).getNumberOfQuizzes();
        }
        return quizNum;
    }

    @Override
    public String toString() {
        String message = name + ",";
        for (Unit unit : units) {
            message = message + unit + ",";
        }
        return message.substring(0, message.length() - 1);

    }
}
