#include <ui/Plotter.hpp>

START_NAMESPACE_UI

constexpr float _MIN = -0x7fffffff, _MAX = 0x7fffffff;
constexpr int axisThickness = 2, pointRadius = 4;

constexpr bool _checkState(int state, int value){
  return (state & value) == value;
}

void do_nothing(pltdata&) {return;}

Plotter::Plotter(
  DrawingPolicy policy, 
  std::size_t maxWidth, 
  std::size_t maxHeight
): _maxWidth(maxWidth), _maxHeight(maxHeight), _xAxisPos(maxHeight - axisThickness), _yAxisPos(0),
_maxValue(_MIN), _minValue(_MAX), _minRange(_MAX), _maxRange(_MIN), 
_state(0), _forceUpdate(true), _policy(policy), _plotdata(), 
_window(sf::VideoMode(maxWidth, maxHeight), "Plotter"), _callb(do_nothing)
{}

void Plotter::prepare(DrawingPolicy policy, std::size_t maxWidth, std::size_t maxHeight){
  _policy = policy;
  _maxWidth  = maxWidth;
  _maxHeight = maxHeight;

  _window.create(sf::VideoMode(maxWidth, maxHeight), "Plotter");
}

void Plotter::values(float min, float max){
  _minValue = min;
  _maxValue = max;
  _state ^= _stateStrictValues;
}

void Plotter::range(float min, float max){
  _minRange = min;
  _maxRange = max;
  _state ^= _stateStrictValues;
}

void Plotter::add(const _PlotPoint& point){
  _rawdata.push_back(point);
}

void Plotter::add(const _rawdatatype& data){

}

void Plotter::update(){
  _forceUpdate = true;
}

void Plotter::addCallback(std::function<void(pltdata&)> callb){
  _callb = callb;
}

void Plotter::open(){

  _window.setActive(true);
  _window.setFramerateLimit(30);
  _window.display();

  while(_window.isOpen()){
    sf::Event e;
    if (_window.pollEvent(e)){
      switch (e.type)
      {
      case sf::Event::Closed:
        _window.close();
        break;
      default:
        break;
      }
    }

    _prepareData();
    _drawBackground();
    _drawAxis();
    _drawData();
    _window.display();

    _callb(_plotdata);
  }
}


bool Plotter::_isStrict(){
  return _checkState(_state, _stateStrictValues);
}

int Plotter::_getNormalizedY(float y, bool policyEffect){
  int ret = static_cast<int>((1 - (y - _minValue) / (_maxValue - _minValue)) * _maxHeight);
  if (policyEffect && _policy == DrawingPolicy::Point){
    ret -= pointRadius*2;
  }
  return ret;
}

int Plotter::_getNormalizedX(float x){
  return static_cast<int>((x - _minRange) / (_maxRange - _minRange) * _maxWidth);
}

_Plot Plotter::_normalize(const _PlotPoint& point){

  if (_isStrict()){
    if (point.x < _minValue || point.x > _maxValue || point.y < _minValue || point.y > _maxValue){
      throw std::runtime_error("Values out of range");
    }
  }
  return {
    _getNormalizedX(point.x),
    _getNormalizedY(point.y)
  };
}

void Plotter::_drawBackground(){
  _window.clear();
}

void Plotter::_drawAxis(){
  using namespace sf;
  RectangleShape xA(Vector2f(_maxWidth, axisThickness));
  RectangleShape yA(Vector2f(axisThickness, _maxHeight));

  xA.setFillColor(sf::Color::Magenta);
  yA.setFillColor(sf::Color::Magenta);

  xA.setPosition(0, _xAxisPos);
  yA.setPosition(_yAxisPos, 0);

  _window.draw(xA);
  _window.draw(yA);
}

void Plotter::_drawPoint(_Plot& current, _Plot& prev){
  using namespace sf;
  switch (_policy)
  {
  case DrawingPolicy::Point:{
    CircleShape p(pointRadius);
    p.setPosition(current.x, current.y);
    p.setFillColor(Color::Blue);
    _window.draw(p);
  }
    break;
  
  default:
    break;
  }
}

void Plotter::_prepareData(){
  if (!_forceUpdate){
    return;
  }

  
  for (size_t i = 0; i < _rawdata.size(); i++){
    _minValue = std::min(_minValue, _rawdata[i].y);
    _maxValue = std::max(_maxValue, _rawdata[i].y);

    _minRange = std::min(_minRange, _rawdata[i].x);
    _maxRange = std::max(_maxRange, _rawdata[i].x);
  }

  if (_minValue == _maxValue){
    _minValue -= 1;
    _maxValue += 1;
  }

  if (_minRange == _maxRange){
    _minRange -= 1;
    _maxRange += 1;
  }

  _xAxisPos = _getNormalizedY(0, false);
  _yAxisPos = abs(_getNormalizedX(0));

  _xAxisPos = std::min(_xAxisPos, static_cast<int>(_maxHeight) - axisThickness);
  _yAxisPos = std::min(_yAxisPos, static_cast<int>(_maxWidth) - axisThickness);

  _plotdata.clear();
  _plotdata.reserve(_rawdata.size());

  for (size_t i = 0; i < _rawdata.size(); i++){
    _plotdata.push_back(_normalize(_rawdata[i]));
  }

  _forceUpdate = false;
}

void Plotter::_drawData(){
  if (_plotdata.empty()){
    return;
  }

  _Plot prev = _plotdata.front();
  for (auto& point : _plotdata){
    _drawPoint(point, prev);
    prev = point;
  }
}

END_NAMESPACE