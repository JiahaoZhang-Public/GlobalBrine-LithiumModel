import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import Landing from "./pages/Landing";
import MapExplorer from "./pages/MapExplorer";
import SinglePrediction from "./pages/SinglePrediction";
import BatchPrediction from "./pages/BatchPrediction";
import ModelPage from "./pages/ModelPage";
import TeamPage from "./pages/TeamPage";

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/map" element={<MapExplorer />} />
        <Route path="/predict" element={<SinglePrediction />} />
        <Route path="/batch" element={<BatchPrediction />} />
        <Route path="/model" element={<ModelPage />} />
        <Route path="/team" element={<TeamPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}

export default App;
