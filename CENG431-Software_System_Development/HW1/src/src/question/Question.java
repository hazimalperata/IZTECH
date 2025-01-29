package src.question;

public abstract class Question implements IQuestion {
    private int points;

    public Question() {
        this(0);
    }

    public Question(int points) {
        this.points = points;
    }

    @Override
    public int getPoints() {
        return points;
    }
}
