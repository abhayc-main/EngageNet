type CardProps = {
  children: React.ReactNode | React.ReactNode[];
};

const Card: React.FC<CardProps> = ({ children }) => {
  return (
    <div className="card">
      {children}
    </div>
  );
};

export default Card;
