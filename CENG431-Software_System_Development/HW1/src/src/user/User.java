package src.user;

import src.language.Language;
import src.league.ILeague;
import src.league.League;
import src.quiz.Quiz;
import src.unit.Unit;
import utils.Rand;

public class User {
    private final int MAX_STREAK_NUMBER = 365;
    private final int MIN_STREAK_NUMBER = 0;
    private final int MIN_REQUIRED_QUIZZES_NUM = 6;

    Rand rand = new Rand();

    private String name;
    private String password;
    private Language chosenLanguage;
    private int numberOfDaysStreak;
    private int totalPoints;
    private ILeague currentLeague;
    private Unit currentUnit;
    private int numberOfSolvedQuizzes;
    private int numberOfRequiredQuizzes;

    public User(String name, String password, Language chosenLanguage) {
        this.name = name;
        this.password = password;
        this.chosenLanguage = chosenLanguage;
        this.totalPoints = 0;
        this.numberOfSolvedQuizzes = 0;
        this.numberOfDaysStreak = rand.getInt(MIN_STREAK_NUMBER, MAX_STREAK_NUMBER);
        this.numberOfRequiredQuizzes = rand.getInt(MIN_REQUIRED_QUIZZES_NUM, chosenLanguage.getNumberOfTotalQuizzes());
        this.currentLeague = chosenLanguage.getBronzeLeague();
        this.currentLeague.appendUser(this);
    }

    public void solveQuizzes() {
        for (Unit unit : chosenLanguage.getUnits()) {
            this.currentUnit = unit;
            if (numberOfRequiredQuizzes != numberOfSolvedQuizzes){
                for (Quiz quiz: unit.getQuizzes()){
                    int quizPoint = quiz.solve();
                    totalPoints += quizPoint;
                    numberOfSolvedQuizzes += 1;
                    if (numberOfRequiredQuizzes == numberOfSolvedQuizzes){
                        break;
                    }
                }
            }else{
                break;
            }
        }
    }

    public String getName() {
        return name;
    }

    public String getPassword() {
        return password;
    }

    public Language getChosenLanguage() {
        return chosenLanguage;
    }

    public int getNumberOfSolvedQuizzes() {
        return numberOfSolvedQuizzes;
    }

    public Unit getCurrentUnit() {
        return currentUnit;
    }

    public int getTotalPoints() {
        return totalPoints;
    }

    public void setChosenLanguage(Language chosenLanguage) {
        this.chosenLanguage = chosenLanguage;
    }

    public ILeague getCurrentLeague() {
        return this.currentLeague;
    }

    public void setCurrentLeague(League currentLeague) {
        this.currentLeague = currentLeague;
    }

    public int getNumberOfDaysStreak() {
        return numberOfDaysStreak;
    }

}
