"""Exercise persistent beam controls with real Qt interactions, including frozen builds."""
import json

from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from .core import ArrayParameters
from .feed import FeedSettings, LEGACY_SETTINGS
from .planar import PlanarParameters


def exercise_pointing(app, window, output, report, check):
    window.ribbon.setCurrentIndex(0)
    feed, planar = window.feed_panel, window.planar_panel
    # The preceding sparse workflow transfers its selected geometry and scan angles.
    check(float(planar.geometry_fields["scan_theta_deg"].text()) == planar.read_parameters().scan_theta_deg
          and float(planar.geometry_fields["scan_phi_deg"].text()) == planar.read_parameters().scan_phi_deg,
          "sparse_handoff_populates_visible_beam_angles")

    def click(widget):
        QTest.mouseClick(widget, Qt.MouseButton.LeftButton); app.processEvents()

    def type_angle(widget, value):
        widget.setFocus(); widget.selectAll(); QTest.keyClicks(widget, str(value)); app.processEvents()

    warnings = []
    def dismiss():
        modal = QApplication.activeModalWidget()
        if isinstance(modal, QMessageBox): warnings.append(modal.text()); modal.accept()

    window.tabs.setCurrentIndex(0);app.processEvents()
    check(window.result_bar.isVisible(), "uniform_array_result_selector_restored_outside_electrical_pages")
    window.tabs.setCurrentIndex(4)
    window.apply_parameters(ArrayParameters(elements=8, spacing=.45, step_deg=1))
    feed.electrical.apply(FeedSettings(phase_mode="none"))
    type_angle(feed.target_theta, -24)
    check(window.phase_mode.currentIndex()==1 and window.read_parameters().scan_deg==-24,
          "typing_feed_target_enables_shared_scan_from_broadside")
    window.inputs["scan_deg"].setText("33")
    check(feed.target_theta.text()=="33", "right_scan_edits_update_persistent_feed_target")
    window.phase_mode.setCurrentIndex(0)
    check(float(feed.target_theta.text())==0, "global_broadside_updates_effective_feed_input")
    type_angle(feed.target_theta,-24)
    for topology in ("corporate","series","symmetric_series"):
        feed.topology.setCurrentIndex(feed.topology.findData(topology))
        feed.electrical.phase_mode.setCurrentIndex(feed.electrical.phase_mode.findData("scan"))
        click(window.compute_button)
        p = feed.report["electrical"]["pointing"]
        check(feed.read_settings().phase_mode=="scan" and abs(p["output_cut_peak_deg"]+24)<1e-7,
              f"{topology}_pointing_button_generates_requested_beam")
    feed.save_path(output/"pointing-linear.json")
    report["pointing_results"]={"linear":feed.report}
    selected=feed.report
    type_angle(feed.target_theta,12)
    check(feed.report is selected and "尚未重新生成" in feed.state.text(), "beam_edit_keeps_previous_result_snapshot")
    feed.electrical.phase_mode.setCurrentIndex(feed.electrical.phase_mode.findData("none"))
    type_angle(feed.target_theta,91)
    QTimer.singleShot(100,dismiss);click(window.compute_button)
    check(len(warnings)==1 and feed.report is selected and feed.read_settings().phase_mode=="none",
          "invalid_beam_angle_preserves_result_and_compensation_mode")
    feed.load_path(output/"pointing-linear.json")
    check(float(feed.target_theta.text())==-24 and window.read_parameters().scan_deg==-24 and feed.report is selected,
          "saved_pointing_restores_both_angle_controls_without_recompute")
    legacy=json.loads((output/"pointing-linear.json").read_text(encoding="utf-8"))
    legacy["schema_version"]=1
    legacy["settings"]={key:value for key,value in legacy["settings"].items() if key in LEGACY_SETTINGS}
    (output/"pointing-legacy.json").write_text(json.dumps(legacy),encoding="utf-8")
    feed.load_path(output/"pointing-legacy.json")
    check(feed.read_settings().phase_mode=="none" and float(feed.target_theta.text())==-24,
          "legacy_feed_load_keeps_uncompensated_mode_and_requested_angle")
    feed.topology.setCurrentIndex(feed.topology.findData("corporate"));click(window.compute_button)
    p=feed.report["electrical"]["pointing"]
    check(p["target_theta_deg"]==-24 and abs(p["output_cut_peak_deg"])<1e-7,
          "normal_generate_preserves_none_mode_and_reports_actual_peak")
    feed.load_path(output/"pointing-linear.json");click(window.compute_button)
    feed.result_tabs.setCurrentIndex(3);window.resize(1440,900);app.processEvents()
    check(feed.beam_row.isVisible() and feed.settings_tabs.isVisible() and feed.parameters_in_sidebar, "feed_beam_controls_remain_in_task_sidebar")
    check("-24" in feed.pattern_metrics.text() and "输出" in feed.pattern_metrics.text(), "feed_pointing_summary_rendered")
    window.grab().save(str(output/"pointing-linear.png"))
    window.resize(1100,740);app.processEvents()
    check(window.compute_button.isVisible() and feed.pattern_plot.height()>140 and not window.result_bar.isVisible(),
          "compact_feed_beam_controls_and_plot_usable")
    window.grab().save(str(output/"pointing-linear-small.png"))

    window.tabs.setCurrentIndex(5)
    planar.apply_configuration(PlanarParameters(nx=2,ny=2,scan_theta_deg=33,scan_phi_deg=47,step_deg=1),FeedSettings(phase_mode="equalize"))
    click(window.compute_button)
    check(planar.report["electrical"]["pointing"]["target_theta_deg"]==0, "planar_normal_generate_keeps_broadside_mode")
    type_angle(planar.geometry_fields["scan_theta_deg"],38)
    type_angle(planar.geometry_fields["scan_phi_deg"],-62)
    planar.electrical.phase_mode.setCurrentIndex(planar.electrical.phase_mode.findData("scan"))
    click(window.compute_button)
    p=planar.report["electrical"]["pointing"]
    check(planar.read_settings().phase_mode=="scan" and p["target_theta_deg"]==38 and p["target_phi_deg"]==-62
          and p["representative_peak_separation_deg"]<1e-7, "rectangular_pointing_button_applies_visible_theta_phi")
    planar.apply_configuration(PlanarParameters(geometry="arbitrary",coordinates=((-0.5,-0.1),(-0.1,0.4),(0.1,-0.3),(0.5,0.2)),
                                               scan_theta_deg=41,scan_phi_deg=126,cut_phi_deg=126,step_deg=1),
                               FeedSettings(topology="symmetric_series",phase_mode="none"))
    planar.electrical.phase_mode.setCurrentIndex(planar.electrical.phase_mode.findData("scan"))
    click(window.compute_button)
    p=planar.report["electrical"]["pointing"]
    check(p["target_theta_deg"]==41 and p["target_phi_deg"]==126 and p["representative_peak_separation_deg"]<1e-7,
          "arbitrary_planar_pointing_uses_imported_coordinates")
    report["pointing_results"]["planar"]=planar.report
    planar.save_path(output/"pointing-planar.json")
    selected=planar.report
    planar.electrical.phase_mode.setCurrentIndex(planar.electrical.phase_mode.findData("none"))
    type_angle(planar.geometry_fields["scan_theta_deg"],91)
    QTimer.singleShot(100,dismiss);click(window.compute_button)
    check(len(warnings)==2 and planar.report is selected and planar.read_settings().phase_mode=="none",
          "invalid_planar_pointing_preserves_report_and_mode")
    planar.load_path(output/"pointing-planar.json")
    check(planar.report is selected and float(planar.geometry_fields["scan_theta_deg"].text())==41,
          "planar_pointing_load_preserves_result_and_restores_visible_angles")
    planar.view.setCurrentIndex(planar.view.findData("map"));window.resize(1440,900);app.processEvents()
    check(planar.beam_row.isVisible() and planar.settings_tabs.isVisible() and planar.parameters_in_sidebar, "planar_beam_controls_remain_in_task_sidebar")
    check("有效目标" in planar.metrics.text() and "代表峰" in planar.metrics.text(), "planar_target_and_nonunique_peak_rendered_separately")
    window.grab().save(str(output/"pointing-planar.png"))
    window.resize(1100,740);app.processEvents()
    check(window.compute_button.isVisible() and planar.canvas.height()>100, "compact_planar_beam_controls_and_plot_usable")
    window.grab().save(str(output/"pointing-planar-small.png"))
